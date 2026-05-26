"""CR-4 reconfigure-lifecycle validation for the LavaSR plugin.

Contract-level (no real model load — the model is large/GPU). Exercises the
substrate's reconfigure delta path in-process with a fake model object:

  1. reconfigure(device flip) -> RELEASE the model (RELOAD_TRIGGER ->
     _release_model) + RE-APPLY config (_apply_config)
  2. on_disable releases (CR-2)

Run from the repo root in the plugin's env:

    conda run -n cjm-media-plugin-lavasr --no-capture-output python tests_manual/test_reconfigure.py
"""
import sys


def main() -> int:
    from cjm_media_plugin_lavasr.plugin import LavaSRProcessingPlugin

    p = LavaSRProcessingPlugin()
    p._apply_config({"device": "cpu"})
    assert p.config.device == "cpu"

    # 1) device trigger: release + re-apply
    p._model = object()
    p.reconfigure({"device": "cpu"}, {"device": "auto"})
    assert p._model is None, "device RELOAD_TRIGGER must fire _release_model"
    assert p.config.device == "auto", "reconfigure must re-apply config (CR-4)"
    print("[1] reconfigure device cpu->auto: model released + applied  OK")

    # 2) on_disable releases (CR-2)
    p._model = object()
    p.on_disable()
    assert p._model is None, "on_disable must release the model"
    print("[2] on_disable: model released  OK")

    print("RECONFIGURE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
