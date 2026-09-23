import json

from relationscope.inference.predictor import resolve_model_id


def test_resolve_model_id_uses_manifest_version(tmp_path):
    bundle = tmp_path / "model"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"model_version": "modernbert-open-set-v1.1"}),
        encoding="utf-8",
    )

    assert resolve_model_id(bundle) == "modernbert-open-set-v1.1"


def test_resolve_model_id_falls_back_to_directory_name(tmp_path):
    bundle = tmp_path / "fallback-model"
    bundle.mkdir()

    assert resolve_model_id(bundle) == "fallback-model"


def test_resolve_model_id_falls_back_for_invalid_manifest(tmp_path):
    bundle = tmp_path / "broken-manifest"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{not-json", encoding="utf-8")

    assert resolve_model_id(bundle) == "broken-manifest"
