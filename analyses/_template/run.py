"""<analysis name>: <one line on what this run produces>."""

from pathlib import Path

from analytics.config import save_run_config


def main(cfg: dict) -> None:
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    # load data -> transform with src/analytics functions -> write outputs and figures
    save_run_config(cfg, out / "run_config.json")
