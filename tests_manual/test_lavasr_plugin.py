"""Direct test script for the LavaSR speech enhancement plugin.

Run from the repo root:
    python tests_manual/test_lavasr_plugin.py              # temp dirs (cleaned up)
    python tests_manual/test_lavasr_plugin.py --keep-output # persistent output dir

Tests:
1. Import and metadata generation
2. Plugin instantiation and config schema
3. Initialize and cleanup lifecycle
4. is_available check
5. get_info on test audio
6. enhance_speech on test audio (raw segment)
7. enhance_speech on vocals file (Demucs output)
"""

import argparse
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Test audio files
TEST_SEGMENT = REPO_ROOT / "test_files" / "segment_000.mp3"
TEST_VOCALS = REPO_ROOT / "test_files" / "segment_000_vocals.flac"

# Output directory for --keep-output mode
OUTPUT_DIR = REPO_ROOT / "test_output"

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir


def test_import_and_metadata():
    """Test imports and metadata generation."""
    print("=" * 60)
    print("Test 1: Import and metadata")
    print("=" * 60)
    from cjm_media_plugin_lavasr.meta import get_plugin_metadata

    metadata = get_plugin_metadata()
    print(json.dumps(metadata, indent=2))
    assert metadata["name"] == "cjm-media-plugin-lavasr"
    assert metadata["type"] == "media-processing"
    assert metadata["resources"]["requires_gpu"] is False
    assert "HF_HOME" in metadata["env_vars"]
    print("  OK")
    print()


def test_config_schema():
    """Test config dataclass and JSON Schema generation."""
    print("=" * 60)
    print("Test 2: Config schema")
    print("=" * 60)
    from cjm_media_plugin_lavasr.plugin import LavaSRPluginConfig
    from cjm_plugin_system.utils.validation import dataclass_to_jsonschema

    schema = dataclass_to_jsonschema(LavaSRPluginConfig)
    print(json.dumps(schema, indent=2))
    assert "model_path" in schema["properties"]
    assert "device" in schema["properties"]
    assert "denoise" in schema["properties"]
    assert "enhance" in schema["properties"]
    assert "batch_mode" in schema["properties"]
    assert schema["properties"]["output_format"]["enum"] == ["wav", "flac", "mp3"]
    print("  OK")
    print()


def test_lifecycle():
    """Test plugin initialization and cleanup."""
    print("=" * 60)
    print("Test 3: Lifecycle (initialize / cleanup)")
    print("=" * 60)
    from cjm_media_plugin_lavasr.plugin import LavaSRProcessingPlugin

    plugin = LavaSRProcessingPlugin()

    # Initialize with defaults
    plugin.initialize()
    print(f"  Config: device={plugin.config.device}, denoise={plugin.config.denoise}")
    print(f"  Data dir: {plugin._data_dir}")
    assert plugin.config is not None
    assert plugin.storage is not None
    assert plugin.config.denoise is True

    # Initialize with custom config
    plugin.initialize({"denoise": False, "batch_mode": False})
    assert plugin.config.denoise is False
    assert plugin.config.batch_mode is False
    print(f"  Re-initialized: denoise={plugin.config.denoise}, batch_mode={plugin.config.batch_mode}")

    # Cleanup
    plugin.cleanup()
    assert plugin._model is None
    print("  Cleanup OK")
    print()


def test_is_available():
    """Test availability check."""
    print("=" * 60)
    print("Test 4: is_available")
    print("=" * 60)
    from cjm_media_plugin_lavasr.plugin import LavaSRProcessingPlugin

    plugin = LavaSRProcessingPlugin()
    available = plugin.is_available()
    print(f"  Available: {available}")
    assert available is True, "LavaSR should be available in this environment"
    print()


def test_get_info():
    """Test get_info action."""
    print("=" * 60)
    print("Test 5: get_info")
    print("=" * 60)
    from cjm_media_plugin_lavasr.plugin import LavaSRProcessingPlugin

    assert TEST_SEGMENT.exists(), f"Test file not found: {TEST_SEGMENT}"

    plugin = LavaSRProcessingPlugin()
    plugin.initialize()

    result = plugin.execute(action="get_info", file_path=str(TEST_SEGMENT))
    print(f"  Result: {json.dumps(result, indent=2)}")
    assert "duration" in result
    assert result["duration"] > 0
    print()

    plugin.cleanup()


def test_enhance_speech_segment():
    """Test speech enhancement on a raw audio segment."""
    print("=" * 60)
    print("Test 6: enhance_speech (raw segment)")
    print("=" * 60)
    from cjm_media_plugin_lavasr.plugin import LavaSRProcessingPlugin

    assert TEST_SEGMENT.exists(), f"Test file not found: {TEST_SEGMENT}"

    plugin = LavaSRProcessingPlugin()
    plugin.initialize()

    with get_output_dir("segment_enhanced") as out_dir:
        result = plugin.execute(
            action="enhance_speech",
            input_path=str(TEST_SEGMENT),
            output_dir=out_dir,
        )
        print(f"  Result: {json.dumps(result, indent=2)}")

        # Verify output
        output_path = Path(result["output_path"])
        assert output_path.exists(), f"Output file not found: {output_path}"
        assert output_path.stat().st_size > 0, "Output file is empty"
        print(f"  Output size: {output_path.stat().st_size:,} bytes")

        # Verify output sample rate is 48kHz
        assert result["output_sample_rate"] == 48000

        # Verify job in database
        job = plugin.storage.get_by_job_id(result["job_id"])
        assert job is not None, "Job not found in database"
        print(f"  Job stored: {job.job_id}")
        print(f"  Job action: {job.action}")

    plugin.cleanup()
    print()


def test_enhance_speech_vocals():
    """Test speech enhancement on Demucs vocals output (typical pipeline)."""
    print("=" * 60)
    print("Test 7: enhance_speech (Demucs vocals)")
    print("=" * 60)
    from cjm_media_plugin_lavasr.plugin import LavaSRProcessingPlugin

    assert TEST_VOCALS.exists(), f"Test file not found: {TEST_VOCALS}"

    plugin = LavaSRProcessingPlugin()
    plugin.initialize()

    with get_output_dir("vocals_enhanced") as out_dir:
        result = plugin.execute(
            action="enhance_speech",
            input_path=str(TEST_VOCALS),
            output_dir=out_dir,
        )
        print(f"  Result: {json.dumps(result, indent=2)}")

        # Verify output
        output_path = Path(result["output_path"])
        assert output_path.exists(), f"Output file not found: {output_path}"
        assert output_path.stat().st_size > 0, "Output file is empty"
        print(f"  Output size: {output_path.stat().st_size:,} bytes")

        # Verify defaults applied
        assert result["denoise_applied"] is True
        assert result["enhance_applied"] is True

    plugin.cleanup()
    print()


def main():
    global KEEP_OUTPUT

    parser = argparse.ArgumentParser(description="LavaSR plugin test suite")
    parser.add_argument("--keep-output", action="store_true",
                        help=f"Save output files to {OUTPUT_DIR}/ instead of temp dirs")
    args = parser.parse_args()
    KEEP_OUTPUT = args.keep_output

    print()
    print("cjm-media-plugin-lavasr Test Suite")
    print("=" * 60)
    if KEEP_OUTPUT:
        print(f"  Output dir: {OUTPUT_DIR}")
    print()

    test_import_and_metadata()
    test_config_schema()
    test_lifecycle()
    test_is_available()
    test_get_info()
    test_enhance_speech_segment()
    test_enhance_speech_vocals()

    print("=" * 60)
    print("All tests passed!")
    if KEEP_OUTPUT:
        print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
