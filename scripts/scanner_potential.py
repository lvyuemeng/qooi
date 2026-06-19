from __future__ import annotations

import argparse
from pathlib import Path

from qooi.scanner.workflow import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the potential scanner")
    parser.add_argument("--config", default="configs/potential-daily-tailtree.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(Path(args.config)))


if __name__ == "__main__":
    main()
