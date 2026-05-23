"""Config-first QOOI research command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qooi.research.config import (
    PROFILE_CHOICES,
    ResearchCommandConfig,
    load_research_command_config,
)
from qooi.research.signal_reports import (
    run_backtest_workflow,
    run_cache_audit,
    run_research_evaluation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QOOI research workflows")
    parser.add_argument("--config", default="")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--exclude-symbol", default=None)
    parser.add_argument("--diagnostics-export", default=None)
    parser.add_argument("--refresh-cache", action="store_true", default=None)
    parser.add_argument("--cache-audit", action="store_true", default=None)
    parser.add_argument("--show-status", action="store_true", default=None)
    return parser.parse_args()


def _command_config(cli_args: argparse.Namespace) -> ResearchCommandConfig:
    config = (
        load_research_command_config(Path(cli_args.config))
        if cli_args.config
        else ResearchCommandConfig()
    )

    run_updates = {}
    cache_updates = {}
    diagnostics_updates = {}
    if cli_args.profile is not None:
        run_updates["profile"] = cli_args.profile
    if cli_args.symbol is not None:
        run_updates["symbol"] = cli_args.symbol
    if cli_args.exclude_symbol is not None:
        run_updates["exclude_symbol"] = tuple(
            item.strip() for item in cli_args.exclude_symbol.split(",") if item.strip()
        )
    if cli_args.diagnostics_export is not None:
        diagnostics_updates["export"] = cli_args.diagnostics_export
    if cli_args.refresh_cache:
        cache_updates["refresh"] = True
    if cli_args.cache_audit:
        cache_updates["audit"] = True
    if cli_args.show_status:
        run_updates["show_status"] = True

    if run_updates:
        config = config.model_copy(update={"run": config.run.model_copy(update=run_updates)})
    if cache_updates:
        config = config.model_copy(update={"cache": config.cache.model_copy(update=cache_updates)})
    if diagnostics_updates:
        config = config.model_copy(
            update={"diagnostics": config.diagnostics.model_copy(update=diagnostics_updates)}
        )
    return ResearchCommandConfig.model_validate(config.model_dump())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    command = _command_config(_parse_args())
    if command.cache.audit:
        print(run_cache_audit(command))
    elif command.diagnostics.mode == "research-evaluation":
        print(run_research_evaluation(command))
    else:
        print(run_backtest_workflow(command))


if __name__ == "__main__":
    main()
