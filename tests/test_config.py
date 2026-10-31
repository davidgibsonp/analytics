import json

import pytest

from analytics.config import apply_overrides, load_config, parse_value, save_run_config


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3", 3),
        ("0.5", 0.5),
        ("true", True),
        ("False", False),
        ('["a","b"]', ["a", "b"]),
        ("kmeans", "kmeans"),
        ("'quoted'", "quoted"),
    ],
)
def test_parse_value(raw, expected):
    assert parse_value(raw) == expected


def test_apply_overrides_sets_known_keys():
    cfg = {"tfidf": {"min_df": 2}}
    assert apply_overrides(cfg, ["tfidf.min_df=3"])["tfidf"]["min_df"] == 3


@pytest.mark.parametrize("bad", ["tfidf.bogus=1", "nosection.x=1", "tfidf.min_df", "min_df=3"])
def test_apply_overrides_rejects_unknown_or_malformed(bad):
    with pytest.raises(ValueError):
        apply_overrides({"tfidf": {"min_df": 2}}, [bad])


def test_load_config_resolves_paths_and_records_overrides(tmp_path):
    (tmp_path / "config.toml").write_text('[paths]\noutputs = "outputs"\n[a]\nb = 1\n')
    cfg = load_config(tmp_path / "config.toml", ["a.b=2"])
    assert cfg["paths"]["outputs"] == str(tmp_path / "outputs")
    assert cfg["a"]["b"] == 2
    assert cfg["_meta"]["overrides"] == ["a.b=2"]
    save_run_config(cfg, tmp_path / "run.json", rows=5)
    assert json.loads((tmp_path / "run.json").read_text())["rows"] == 5
