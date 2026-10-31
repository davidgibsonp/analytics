"""Config loading: a TOML file plus `--set section.key=value` overrides.

Every tunable value of an analysis lives in its config.toml. Overrides must name an existing
key so a typo fails loudly instead of being ignored. Each run saves the exact config it used.
"""

from __future__ import annotations

import ast
import json
import os
import tomllib
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
ANALYSES_DIR = REPO_ROOT / "analyses"
DATA_DIR = REPO_ROOT / "data"


def parse_value(raw: str) -> Any:
    """Parse an override value: Python/TOML literals, else the bare string."""
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `section.key=value` overrides in place. Unknown keys raise ValueError."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        if not sep or not name:
            raise ValueError(f"override {item!r} must look like section.key=value")
        if section not in cfg or not isinstance(cfg[section], dict) or name not in cfg[section]:
            raise ValueError(f"override {item!r}: unknown key {key!r}")
        cfg[section][name] = parse_value(raw)
    return cfg


def load_config(path: Path | str, overrides: list[str] | None = None) -> dict:
    """Load a TOML config, apply overrides, and resolve [paths] relative to the config file."""
    path = Path(path).resolve()
    cfg = tomllib.loads(path.read_text())
    apply_overrides(cfg, overrides or [])
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    cfg["_meta"] = {"config_path": str(path), "overrides": list(overrides or [])}
    return cfg


def save_run_config(cfg: dict, path: Path | str, **extra: Any) -> None:
    """Write the config a run used (plus any extra facts) as JSON next to its outputs."""
    Path(path).write_text(json.dumps({**cfg, **extra}, indent=2, default=str))


def load_env(path: Path | str = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing values."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
