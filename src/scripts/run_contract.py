"""Lightweight pre-scoring bootstrap for a future canonical runner."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Generic, TypeVar

from src.utils.provenance import ProvenanceError, RunIdentity, build_run_identity


CANONICAL_MASTER_SEED = 2025
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class PreScoringExecution(Generic[ResultT]):
    """Identity created before, and result returned by, a scoring callback."""

    identity: RunIdentity
    result: ResultT


def execute_pre_scoring(
    *,
    master_seed: int,
    identity_inputs: Mapping[str, Any],
    scoring_stage: Callable[[RunIdentity], ResultT],
) -> PreScoringExecution[ResultT]:
    """Build the immutable identity before invoking a supplied scoring stage."""

    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, Integral)
        or master_seed < 0
    ):
        raise ProvenanceError("master seed must be a non-negative integer")
    if not isinstance(identity_inputs, Mapping):
        raise ProvenanceError("identity inputs must be a mapping")
    if not callable(scoring_stage):
        raise ProvenanceError("scoring stage must be callable")

    prepared = dict(identity_inputs)
    raw_task = prepared.get("task")
    if not isinstance(raw_task, Mapping):
        raise ProvenanceError("task must be a mapping")
    task = dict(raw_task)
    declared_seed = task.get("master_seed")
    if "master_seed" in task and declared_seed != int(master_seed):
        raise ProvenanceError(
            "task master seed does not match the explicitly supplied master seed"
        )
    task["master_seed"] = int(master_seed)
    prepared["task"] = task

    identity = build_run_identity(**prepared)
    result = scoring_stage(identity)
    return PreScoringExecution(identity=identity, result=result)
