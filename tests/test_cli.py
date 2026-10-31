import shutil

import pytest

from analytics import cli


@pytest.fixture
def analyses(tmp_path, monkeypatch):
    """A temporary analyses/ folder with the template and a copy of it as `demo`."""
    shutil.copytree(cli.ANALYSES_DIR / "_template", tmp_path / "_template")
    shutil.copytree(cli.ANALYSES_DIR / "_template", tmp_path / "demo")
    monkeypatch.setattr(cli, "ANALYSES_DIR", tmp_path)
    return tmp_path


def test_run_template_with_override(analyses):
    cli.main(["run", "demo", "--set", "data.snapshot=2026-01-01"])
    assert (analyses / "demo" / "outputs" / "run_config.json").exists()


def test_run_rejects_unknown_override(analyses):
    with pytest.raises(SystemExit, match="unknown key"):
        cli.main(["run", "demo", "--set", "data.nope=1"])


def test_new_copies_template(analyses):
    cli.main(["new", "second"])
    assert (analyses / "second" / "run.py").exists()


def test_list_skips_template(analyses, capsys):
    cli.main(["list"])
    out = capsys.readouterr().out
    assert "demo" in out and "_template" not in out
