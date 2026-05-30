"""LavaSR Phase-3-bundle end-to-end validation (GPU).

Validates the cjm-torch-plugin-utils adoption (release_model + cuda_oom +
resolve_torch_device), the cjm-hf-plugin-utils HFCacheConfig + sentinel-bypass
snapshot_download_with_progress, the Shape-2 heartbeat around the LavaEnhance2
constructor load, the Q3 Layer B cache_dir_for_config output dir, and the Track 19
WORKER_ENV migration live — mirroring the Demucs / Voxtral-HF Phase 3 validation.

It also doubles as an A/B harness for the enhancement-quality question: the input is
first converted to a clean PCM wav via ffmpeg.convert, then handed to LavaSR. Tune the
ffmpeg conversion knobs below (or swap INPUT_AUDIO to the vocals flac) and re-run to
compare. The enhanced output path is logged prominently for listening.

Run from the lavasr repo root after:
  1. `cjm-ctl --cjm-config cjm.yaml setup-runtime`
  2. `cjm-ctl --cjm-config cjm.yaml install-all --plugins plugins_test.yaml`
     (lavasr + ffmpeg + cjm-system-monitor-nvidia)
  3. A speech clip at test_files/segment_000.mp3 (alt: segment_000_vocals.flac)

Then:
  conda run -n cjm-media-plugin-lavasr --no-capture-output \\
    python tests_manual/validate_lavasr_e2e.py
"""
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("lavasr-e2e")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── A/B knobs (edit + re-run to compare enhancement quality) ──────────────────
# Input audio fed into the ffmpeg -> lavasr sequence. The .mp3 is the raw segment;
# the _vocals.flac is the demucs-separated speech (the realistic upstream input).
INPUT_AUDIO = REPO_ROOT / "test_files" / "segment_000.mp3"
# ffmpeg.convert knobs. Setting CONVERT_SAMPLE_RATE = None keeps the native rate
# (LavaSR resamples to input_sr internally either way); CONVERT_CHANNELS = 1 downmixes
# to mono (speech). The hypothesis under test: a clean PCM-wav decode improves the
# (previously underwhelming) LavaSR output vs feeding the compressed file directly.
CONVERT_OUTPUT_FORMAT = "wav"
CONVERT_SAMPLE_RATE: Optional[int] = None
CONVERT_CHANNELS: Optional[int] = 1
# ──────────────────────────────────────────────────────────────────────────────

MANIFESTS_DIR = REPO_ROOT / ".cjm" / "manifests"
EMPIRICAL_DB = REPO_ROOT / ".cjm" / "empirical_resources.db"

PLUGIN_NAME = "cjm-media-plugin-lavasr"
SYSMON_NAME = "cjm-system-monitor-nvidia"
FFMPEG_NAME = "cjm-media-plugin-ffmpeg"


def check_prereqs() -> None:
    assert INPUT_AUDIO.exists(), f"Missing test audio: {INPUT_AUDIO}"
    assert MANIFESTS_DIR.exists(), (
        f"Missing manifests dir: {MANIFESTS_DIR} — run cjm-ctl setup-runtime + install-all first"
    )
    for name in (PLUGIN_NAME, SYSMON_NAME, FFMPEG_NAME):
        assert (MANIFESTS_DIR / f"{name}.json").exists(), f"Missing manifest: {name}.json"
    log.info("Prereqs OK: test audio + lavasr + nvidia-monitor + ffmpeg manifests present")


def assert_manifest_shape() -> None:
    manifest = json.loads((MANIFESTS_DIR / f"{PLUGIN_NAME}.json").read_text())
    assert manifest["format_version"] == "2.0", manifest["format_version"]
    code = manifest["code"]

    desc = code.get("description") or manifest.get("description") or ""
    assert desc.strip(), "manifest description is empty (T24 regression)"
    log.info(f"Manifest T24 description: {desc!r}")

    tax = code["taxonomy"]
    assert tax["domain"] == "media" and tax["role"] == "MediaProcessingPlugin", tax
    # LavaSR is GPU-OPTIONAL (CPU path exists) -> requires_gpu is False by design.
    assert code["resources"].get("requires_gpu") is False, code["resources"]
    for stale in ("min_gpu_vram_mb", "recommended_gpu_vram_mb", "min_system_ram_mb"):
        assert stale not in code["resources"], f"stale resource field present: {stale}"
    log.info(f"Manifest CR-1/Phase-5a: taxonomy={tax}, resources={code['resources']}")

    # Track 19: CUDA_VISIBLE_DEVICES (static) + HF_HOME (templated); empty install.env_vars.
    worker_env = code.get("worker_env", [])
    by_name = {e["name"]: e for e in worker_env}
    assert {"CUDA_VISIBLE_DEVICES", "HF_HOME"} <= set(by_name), (
        f"Track 19 WORKER_ENV missing expected vars: {sorted(by_name)}"
    )
    hf_home_default = by_name["HF_HOME"].get("default", "")
    assert hf_home_default == "${CJM_MODELS_DIR}/huggingface", (
        f"HF_HOME default not templated: {hf_home_default!r}"
    )
    install_env = manifest.get("install", {}).get("env_vars", {})
    assert not install_env, f"install.env_vars should be empty post-migration: {install_env}"
    log.info(f"Manifest Track 19 worker_env: {sorted(by_name)} | HF_HOME default={hf_home_default!r}; install.env_vars empty")


def run_e2e() -> None:
    import asyncio

    from cjm_plugin_system.core.manager import PluginManager
    from cjm_plugin_system.core.config import get_config
    from cjm_plugin_system.core.queue import JobQueue, SequenceStep, JobStatus

    cfg = get_config()
    log.info(f"data_dir={cfg.data_dir}, models_dir={cfg.models_dir}")

    pm = PluginManager(search_paths=[MANIFESTS_DIR], sysmon_plugin_name=SYSMON_NAME)
    pm.discover_manifests()
    log.info(f"Discovered: {[m.name for m in pm.discovered]}")

    pm.load_plugin(next(m for m in pm.discovered if m.name == SYSMON_NAME))
    pm.load_plugin(next(m for m in pm.discovered if m.name == FFMPEG_NAME))
    lavasr_meta = next(m for m in pm.discovered if m.name == PLUGIN_NAME)
    db_path = lavasr_meta.manifest.get("db_path")
    ok = pm.load_plugin(lavasr_meta, config={})
    assert ok, f"Failed to load {PLUGIN_NAME}"
    lavasr_id = lavasr_meta.name
    log.info(f"Loaded {SYSMON_NAME} + {FFMPEG_NAME} + {PLUGIN_NAME}; db_path={db_path}")

    # CR-4 prefetch: LavaEnhance2 constructor downloads HF weights (cold cache) via the
    # sentinel-bypass snapshot_download_with_progress, wrapped by the substrate heartbeat.
    log.info("Calling prefetch() to download + load the LavaSR model...")
    t0 = time.time()
    pm.get_plugin(lavasr_id).prefetch()
    log.info(f"prefetch() returned in {time.time() - t0:.1f}s")

    # ffmpeg.convert writes to <ffmpeg_data_dir>/converted/<stem>.<fmt>.
    ffmpeg_data_dir = Path(next(m for m in pm.discovered if m.name == FFMPEG_NAME).manifest["db_path"]).parent
    predicted_wav = ffmpeg_data_dir / "converted" / f"{INPUT_AUDIO.stem}.{CONVERT_OUTPUT_FORMAT}"
    log.info(f"ffmpeg will convert {INPUT_AUDIO.name} -> {predicted_wav} "
             f"(sample_rate={CONVERT_SAMPLE_RATE}, channels={CONVERT_CHANNELS})")

    convert_kwargs = {
        "action": "convert", "input_path": str(INPUT_AUDIO),
        "output_format": CONVERT_OUTPUT_FORMAT,
    }
    if CONVERT_SAMPLE_RATE is not None:
        convert_kwargs["sample_rate"] = CONVERT_SAMPLE_RATE
    if CONVERT_CHANNELS is not None:
        convert_kwargs["channels"] = CONVERT_CHANNELS

    async def run_sequence() -> Any:
        queue = JobQueue(deps=pm, sysmon_plugin_name=SYSMON_NAME)
        await queue.start()
        try:
            seq_id = await queue.submit_sequence(
                steps=[
                    SequenceStep(plugin_instance_id=FFMPEG_NAME, kwargs=convert_kwargs),
                    SequenceStep(plugin_instance_id=lavasr_id, kwargs={
                        "action": "enhance_speech", "input_path": str(predicted_wav),
                    }),
                ],
                fail_fast=True,
            )
            log.info(f"Submitted sequence {seq_id}: ffmpeg.convert -> lavasr.enhance_speech")
            terminal = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
            while True:
                seq = queue.get_sequence(seq_id)
                if seq is None:
                    raise RuntimeError(f"sequence {seq_id} disappeared")
                if seq.status in terminal:
                    break
                await asyncio.sleep(0.5)
            if seq.status != JobStatus.completed:
                raise RuntimeError(f"Sequence {seq_id} status={seq.status}; results={seq.results}")
            return seq.results
        finally:
            await queue.stop()

    log.info(f"Submitting submit_sequence for {INPUT_AUDIO.name}...")
    t0 = time.time()
    results = asyncio.run(run_sequence())
    log.info(f"Sequence completed in {time.time() - t0:.1f}s")

    # Step 1 (ffmpeg) result: log the converted wav so the user can compare input vs output.
    conv = results[0].result
    conv_path = conv.get("output_path") if isinstance(conv, dict) else getattr(conv, "output_path", None)
    log.info(f"CONVERTED INPUT (listen): {conv_path}")

    # Step 2 (lavasr) result.
    result = results[-1].result
    out_path = result.get("output_path") if isinstance(result, dict) else getattr(result, "output_path", None)
    assert out_path and Path(out_path).exists(), f"enhanced output missing: {out_path} (result={result!r})"
    # Q3 Layer B: output dir is the content+config-addressed cache dir.
    assert "enhance_speech" in out_path, f"output not under cache_dir_for_config layout: {out_path}"
    log.info("=" * 70)
    log.info(f"ENHANCED OUTPUT (listen + A/B vs the converted input above):\n  {out_path}")
    log.info(f"  duration={result.get('duration'):.1f}s, denoise={result.get('denoise_applied')}, "
             f"enhance={result.get('enhance_applied')}, out_sr={result.get('output_sample_rate')}")
    log.info("=" * 70)

    # Plugin DB: confirm the job row persisted.
    if db_path and Path(db_path).exists():
        con = sqlite3.connect(db_path)
        try:
            for t in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                log.info(f"plugin DB {t}: {n} rows")
        finally:
            con.close()

    # Empirical store: LavaSR runs device=auto -> cuda on this box, so assert a NON-ZERO
    # gpu peak (real subtree GPU attribution via nvidia-monitor). requires_gpu being False
    # is about the HARD gate (a CPU path exists), independent of the runtime device here.
    assert EMPIRICAL_DB.exists(), f"empirical store not created: {EMPIRICAL_DB}"
    con = sqlite3.connect(EMPIRICAL_DB)
    gpu_peak = 0.0
    try:
        for t in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
            if "gpu_memory_mb_peak_max" not in cols:
                continue
            for r in con.execute(
                f"SELECT * FROM {t} WHERE plugin_name=? OR instance_id=? OR instance_id LIKE ?",
                (PLUGIN_NAME, lavasr_id, f"{PLUGIN_NAME}%"),
            ).fetchall():
                row = dict(zip(cols, r))
                log.info(f"  empirical {t}: {row}")
                gpu_peak = max(gpu_peak, float(row.get("gpu_memory_mb_peak_max") or 0.0))
    finally:
        con.close()
    assert gpu_peak > 0.0, "empirical gpu_memory_mb_peak is 0 — subtree GPU attribution failed (LavaSR ran device=auto→cuda)"
    log.info(f"GPU attribution VERIFIED: lavasr gpu_memory_mb_peak_max={gpu_peak:.1f} MB")

    pm.unload_plugin(lavasr_id)
    pm.unload_plugin(FFMPEG_NAME)
    pm.unload_plugin(SYSMON_NAME)
    log.info("Unloaded plugins; validation done.")


def main() -> int:
    check_prereqs()
    assert_manifest_shape()
    run_e2e()
    log.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
