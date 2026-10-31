"""The single entry point: `uv run analytics <command>`.

analytics list                         list analyses
analytics new <name>                   start an analysis from analyses/_template
analytics run <name> [--set k=v ...]   run analyses/<name>/run.py with its config.toml
analytics fetch <source> [options]     snapshot raw data from a source into data/raw/
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from analytics.config import ANALYSES_DIR, load_config, load_env

# source name -> "module:function". Each fetch function takes parsed CLI args.
SOURCES: dict[str, str] = {
    "linear": "analytics.sources.linear:fetch",
}


def analysis_dirs() -> list[Path]:
    return sorted(p for p in ANALYSES_DIR.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))


def summary(readme: Path) -> str:
    """First non-heading line of a README, used as the one-line description."""
    if not readme.exists():
        return ""
    for line in readme.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            return line.strip()
    return ""


def cmd_list(_: argparse.Namespace) -> None:
    for d in analysis_dirs():
        print(f"{d.name:28} {summary(d / 'README.md')}")


def cmd_new(args: argparse.Namespace) -> None:
    dest = ANALYSES_DIR / args.name
    if dest.exists():
        sys.exit(f"{dest} already exists")
    shutil.copytree(ANALYSES_DIR / "_template", dest)
    print(f"created {dest.relative_to(ANALYSES_DIR.parent)}; fill in README.md, config.toml and run.py")


def cmd_run(args: argparse.Namespace) -> None:
    folder = ANALYSES_DIR / args.name
    if not (folder / "run.py").exists():
        names = ", ".join(d.name for d in analysis_dirs())
        sys.exit(f"no analysis {args.name!r} (available: {names})")
    try:
        cfg = load_config(args.config or folder / "config.toml", args.set)
    except ValueError as e:
        sys.exit(str(e))
    spec = importlib.util.spec_from_file_location(f"analyses.{args.name}.run", folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(cfg)


def cmd_fetch(args: argparse.Namespace) -> None:
    module_name, func = SOURCES[args.source].split(":")
    getattr(importlib.import_module(module_name), func)(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="analytics", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list analyses").set_defaults(func=cmd_list)

    p = sub.add_parser("new", help="start an analysis from the template")
    p.add_argument("name")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("run", help="run an analysis")
    p.add_argument("name")
    p.add_argument("--config", help="use this config file instead of the analysis's config.toml")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value for this run (repeatable)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("fetch", help="snapshot raw data from a source")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--teams", nargs="*", default=[], help="Linear: team keys to fetch (default all)")
    p.add_argument("--include-archived", action="store_true", help="Linear: include archived issues")
    p.set_defaults(func=cmd_fetch)
    return ap


def main(argv: list[str] | None = None) -> None:
    load_env()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
