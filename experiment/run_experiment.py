from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stability import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cognitive stability experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run(cfg)


if __name__ == "__main__":
    main()
