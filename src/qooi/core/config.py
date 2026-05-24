"""Shared strict Pydantic configuration base types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
