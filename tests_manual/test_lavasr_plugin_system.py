"""
Integration Test: LavaSR Processing Plugin via PluginManager

Verifies that the LavaSR plugin can be loaded and executed via JobQueue
over the process boundary through the plugin system.

Run from the cjm-media-plugin-lavasr conda environment:
    python tests_manual/test_lavasr_plugin_system.py
    python tests_manual/test_lavasr_plugin_system.py --keep-output
"""

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from cjm_plugin_system.core.manager import PluginManager
from cjm_plugin_system.core.queue import JobQueue, JobStatus
from cjm_plugin_system.core.scheduling import QueueScheduler


PLUGIN_NAME = "cjm-media-plugin-lavasr"

REPO_ROOT = Path(__file__).parent.parent
TEST_SEGMENT = str(REPO_ROOT / "test_files" / "segment_000.mp3")
TEST_VOCALS = str(REPO_ROOT / "test_files" / "segment_000_vocals.flac")
OUTPUT_DIR = REPO_ROOT / "test_output" / "plugin_system"

# Global flag
KEEP_OUTPUT = False


@contextmanager
def get_output_dir(name):
    """Yield an output directory — persistent or temporary based on KEEP_OUTPUT."""
    if KEEP_OUTPUT:
        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        yield str(out_dir)
    else:
        tmp_dir = tempfile.mkdtemp(prefix=f"lavasr_ps_{name}_")
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _reload_plugin(manager: PluginManager, config: dict = None):
    """Unload, re-discover, and reload the plugin with the given config."""
    manager.unload_all()
    manager.discover_manifests()
    plugin_meta = next(item for item in manager.discovered if item.name == PLUGIN_NAME)
    manager.load_plugin(plugin_meta, config or {})


async def test_discover_and_load():
    """Verify the plugin is discovered and loads via PluginManager."""
    print("=" * 60)
    print("TEST: Discover and Load via PluginManager")
    print("=" * 60)

    manager = PluginManager(scheduler=QueueScheduler())
    manager.discover_manifests()

    plugin_meta = next((item for item in manager.discovered if item.name == PLUGIN_NAME), None)
    if not plugin_meta:
        print(f"  Plugin {PLUGIN_NAME} not found in discovered manifests.")
        print("  Have you run 'cjm-ctl install' for this plugin?")
        return None
    print(f"  Discovered: {plugin_meta.name} v{plugin_meta.version}")

    if not manager.load_plugin(plugin_meta, {}):
        print("  Failed to load plugin.")
        return None
    print("  Loaded successfully")

    proxy = manager.plugins.get(PLUGIN_NAME)
    assert proxy is not None, "Plugin proxy not found after loading"
    print(f"  Proxy available: {PLUGIN_NAME}")

    print("  PASSED\n")
    return manager


async def test_get_info_via_queue(manager: PluginManager):
    """Verify get_info works via JobQueue over process boundary."""
    print("=" * 60)
    print("TEST: get_info via JobQueue")
    print("=" * 60)

    if not os.path.exists(TEST_SEGMENT):
        print(f"  SKIPPED — {TEST_SEGMENT} not found\n")
        return

    _reload_plugin(manager)

    queue = JobQueue(manager)
    await queue.start()

    job_id = await queue.submit(PLUGIN_NAME, action="get_info", file_path=TEST_SEGMENT, priority=10)
    job = await queue.wait_for_job(job_id, timeout=30)

    assert job.status == JobStatus.completed, f"Expected completed, got {job.status}: {job.error}"
    result = job.result
    assert isinstance(result, dict)
    assert result["path"] == TEST_SEGMENT
    assert result["duration"] > 0
    assert result["size_bytes"] > 0
    assert len(result["audio_streams"]) >= 1
    print(f"  Duration: {result['duration']:.1f}s")
    print(f"  Format: {result['format']}")

    await queue.stop()
    print("  PASSED\n")


async def test_enhance_speech_via_queue(manager: PluginManager):
    """Verify enhance_speech works via JobQueue over process boundary."""
    print("=" * 60)
    print("TEST: enhance_speech via JobQueue (raw segment)")
    print("=" * 60)

    if not os.path.exists(TEST_SEGMENT):
        print(f"  SKIPPED — {TEST_SEGMENT} not found\n")
        return

    with get_output_dir("segment_enhanced") as out_dir:
        _reload_plugin(manager)

        queue = JobQueue(manager)
        await queue.start()

        job_id = await queue.submit(
            PLUGIN_NAME,
            action="enhance_speech",
            input_path=TEST_SEGMENT,
            output_dir=out_dir,
            priority=10
        )
        # LavaSR is fast (~4000x realtime GPU), but include headroom for model download + CPU
        job = await queue.wait_for_job(job_id, timeout=300)

        assert job.status == JobStatus.completed, f"Expected completed, got {job.status}: {job.error}"
        result = job.result
        assert "job_id" in result
        assert "output_path" in result
        assert "duration" in result
        assert result["output_sample_rate"] == 48000
        assert result["denoise_applied"] is True
        assert result["enhance_applied"] is True
        assert os.path.exists(result["output_path"])

        file_size = os.path.getsize(result["output_path"])
        print(f"  Output: {os.path.basename(result['output_path'])}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Duration: {result['duration']:.1f}s")
        print(f"  Sample rate: {result['output_sample_rate']}Hz")

        await queue.stop()
        print("  PASSED\n")


async def test_enhance_vocals_via_queue(manager: PluginManager):
    """Verify enhance_speech on Demucs vocals via JobQueue."""
    print("=" * 60)
    print("TEST: enhance_speech via JobQueue (Demucs vocals)")
    print("=" * 60)

    if not os.path.exists(TEST_VOCALS):
        print(f"  SKIPPED — {TEST_VOCALS} not found\n")
        return

    with get_output_dir("vocals_enhanced") as out_dir:
        _reload_plugin(manager)

        queue = JobQueue(manager)
        await queue.start()

        job_id = await queue.submit(
            PLUGIN_NAME,
            action="enhance_speech",
            input_path=TEST_VOCALS,
            output_dir=out_dir,
            priority=10
        )
        job = await queue.wait_for_job(job_id, timeout=300)

        assert job.status == JobStatus.completed, f"Expected completed, got {job.status}: {job.error}"
        result = job.result
        assert os.path.exists(result["output_path"])
        assert result["output_sample_rate"] == 48000

        file_size = os.path.getsize(result["output_path"])
        print(f"  Output: {os.path.basename(result['output_path'])}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Duration: {result['duration']:.1f}s")

        await queue.stop()
        print("  PASSED\n")


async def test_unknown_action_via_queue(manager: PluginManager):
    """Verify unknown action fails correctly via JobQueue."""
    print("=" * 60)
    print("TEST: Unknown action via JobQueue")
    print("=" * 60)

    _reload_plugin(manager)

    queue = JobQueue(manager)
    await queue.start()

    job_id = await queue.submit(PLUGIN_NAME, action="unknown_action", priority=10)
    job = await queue.wait_for_job(job_id, timeout=30)

    assert job.status == JobStatus.failed
    print(f"  Correctly failed: {job.error}")

    await queue.stop()
    print("  PASSED\n")


async def run_integration():
    print()
    if KEEP_OUTPUT:
        print(f"Output dir: {OUTPUT_DIR}")
        print()

    manager = await test_discover_and_load()
    if manager is None:
        print("Aborting — plugin not available.")
        sys.exit(1)

    await test_get_info_via_queue(manager)
    await test_enhance_speech_via_queue(manager)
    await test_enhance_vocals_via_queue(manager)
    await test_unknown_action_via_queue(manager)

    manager.unload_all()
    print("=" * 60)
    print("ALL PLUGIN SYSTEM TESTS PASSED")
    if KEEP_OUTPUT:
        print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LavaSR plugin system integration tests")
    parser.add_argument("--keep-output", action="store_true",
                        help=f"Save output files to {OUTPUT_DIR}/ instead of temp dirs")
    args = parser.parse_args()
    KEEP_OUTPUT = args.keep_output

    asyncio.run(run_integration())
