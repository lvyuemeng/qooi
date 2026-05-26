"""Run classifier-state research evaluation from config."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qooi.research.config import load_research_command_config
from qooi.research.reports import classifier_state_research


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifier-state research evaluation")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    command = load_research_command_config(Path(_parse_args().config))
    if command.diagnostics.mode != "research-evaluation":
        raise SystemExit("classifier_states requires diagnostics.mode='research-evaluation'")
    print(classifier_state_research(command))


if __name__ == "__main__":
    main()
