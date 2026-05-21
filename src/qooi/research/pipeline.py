"""Uniform pipeline contract for config-first research workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from qooi.research.config import ResearchCommandConfig
from qooi.research.run import (
    run_backtest_workflow,
    run_cache_audit,
    run_research_evaluation,
)


@dataclass(frozen=True)
class ResearchArtifact:
    name: str
    value: object


@dataclass(frozen=True)
class ResearchResult:
    text: str = ""
    artifacts: tuple[ResearchArtifact, ...] = ()


@dataclass(frozen=True)
class ResearchContext:
    command: ResearchCommandConfig
    result: ResearchResult | None = None
    artifacts: tuple[ResearchArtifact, ...] = ()


class WorkflowStage(Protocol):
    name: str

    def run(self, context: ResearchContext) -> ResearchContext: ...


@dataclass(frozen=True)
class FunctionStage:
    name: str
    action: Callable[[ResearchContext], ResearchContext]

    def run(self, context: ResearchContext) -> ResearchContext:
        return self.action(context)


@dataclass(frozen=True)
class ResearchWorkflowPlan:
    name: str
    stages: tuple[WorkflowStage, ...]
    render: Callable[[ResearchResult], str]


def _compute_text(name: str, action: Callable[[ResearchCommandConfig], str]) -> FunctionStage:
    def run(context: ResearchContext) -> ResearchContext:
        return ResearchContext(
            command=context.command,
            result=ResearchResult(text=action(context.command)),
            artifacts=context.artifacts,
        )

    return FunctionStage(name=name, action=run)


def _single_stage_plan(
    name: str,
    action: Callable[[ResearchCommandConfig], str],
) -> ResearchWorkflowPlan:
    return ResearchWorkflowPlan(
        name=name,
        render=render_research_result,
        stages=(
            FunctionStage("prepare", lambda context: context),
            _compute_text("compute", action),
            FunctionStage("assemble", lambda context: context),
        ),
    )


def build_backtest_plan(_command: ResearchCommandConfig) -> ResearchWorkflowPlan:
    return _single_stage_plan("backtest", run_backtest_workflow)


def build_cache_audit_plan(_command: ResearchCommandConfig) -> ResearchWorkflowPlan:
    return _single_stage_plan("cache-audit", run_cache_audit)


def build_research_evaluation_plan(_command: ResearchCommandConfig) -> ResearchWorkflowPlan:
    return _single_stage_plan("research-evaluation", run_research_evaluation)


WORKFLOW_BUILDERS: dict[str, Callable[[ResearchCommandConfig], ResearchWorkflowPlan]] = {
    "backtest": build_backtest_plan,
    "research-evaluation": build_research_evaluation_plan,
}


def compile_research_plan(command: ResearchCommandConfig) -> ResearchWorkflowPlan:
    if command.cache.audit:
        return build_cache_audit_plan(command)
    return WORKFLOW_BUILDERS[command.diagnostics.mode](command)


def execute_research_plan(
    plan: ResearchWorkflowPlan,
    command: ResearchCommandConfig,
) -> ResearchResult:
    context = ResearchContext(command=command)
    for stage in plan.stages:
        context = stage.run(context)
    if context.result is None:
        return ResearchResult(artifacts=context.artifacts)
    return context.result


def render_research_result(result: ResearchResult) -> str:
    return result.text
