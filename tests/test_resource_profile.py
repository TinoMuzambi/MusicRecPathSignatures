"""MR-08 process-tree RSS and blockwise similarity tests."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

import src.utils.timing as timing
from src.scripts.compute_similarity import (
    SimilarityComputationError,
    blockwise_cosine_top_k,
)


class _FakeProcess:
    def __init__(
        self,
        *,
        pid,
        rss=4 * 1024 * 1024,
        children=(),
        children_error=None,
        memory_error=None,
    ):
        self.pid = pid
        self._rss = rss
        self._children = list(children)
        self._children_error = children_error
        self._memory_error = memory_error

    def children(self, *, recursive):
        assert recursive is True
        if self._children_error is not None:
            raise self._children_error
        return list(self._children)

    def memory_info(self):
        if self._memory_error is not None:
            raise self._memory_error
        return SimpleNamespace(rss=self._rss)


def _measurement_unavailable_type():
    exception_type = getattr(timing, "ResourceMeasurementUnavailable", None)
    assert isinstance(exception_type, type), (
        "ResourceMeasurementUnavailable must be defined"
    )
    assert issubclass(exception_type, RuntimeError)
    return exception_type


def _wait_for_monitor_thread_to_stop(monitor):
    deadline = time.monotonic() + 2.0
    while monitor.is_running and time.monotonic() < deadline:
        time.sleep(0.005)
    assert monitor.is_running is False


def test_root_process_construction_failure_is_explicit(monkeypatch):
    unavailable = _measurement_unavailable_type()

    def denied_root():
        raise psutil.AccessDenied(pid=101)

    monkeypatch.setattr(timing.psutil, "Process", denied_root)
    with pytest.raises(unavailable) as caught:
        timing.process_tree_rss_mb()

    assert caught.value.stage == "root_process"
    assert caught.value.cause_type == "AccessDenied"


def test_recursive_child_enumeration_failure_is_explicit():
    unavailable = _measurement_unavailable_type()
    root = _FakeProcess(
        pid=101,
        children_error=psutil.AccessDenied(pid=101),
    )

    with pytest.raises(unavailable) as caught:
        timing.process_tree_rss_mb(root)

    assert caught.value.stage == "child_enumeration"
    assert caught.value.cause_type == "AccessDenied"


def test_root_memory_failure_is_explicit():
    unavailable = _measurement_unavailable_type()
    root = _FakeProcess(
        pid=101,
        memory_error=psutil.AccessDenied(pid=101),
    )

    with pytest.raises(unavailable) as caught:
        timing.process_tree_rss_mb(root)

    assert caught.value.stage == "root_rss"
    assert caught.value.cause_type == "AccessDenied"


def test_disappearing_child_is_tolerated_but_denied_child_is_not():
    unavailable = _measurement_unavailable_type()
    gone = _FakeProcess(
        pid=202,
        memory_error=psutil.NoSuchProcess(pid=202),
    )
    assert timing.process_tree_rss_mb(
        _FakeProcess(pid=101, children=(gone,))
    ) == 4.0

    denied = _FakeProcess(
        pid=203,
        memory_error=psutil.AccessDenied(pid=203),
    )
    with pytest.raises(unavailable) as caught:
        timing.process_tree_rss_mb(
            _FakeProcess(pid=101, children=(denied,))
        )

    assert caught.value.stage == "child_rss"
    assert caught.value.cause_type == "AccessDenied"


def test_background_measurement_failure_is_latched_and_cannot_recover():
    unavailable = _measurement_unavailable_type()
    calls = []
    second_call = threading.Event()

    def sampler():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return 10.0
        if len(calls) == 2:
            second_call.set()
            raise unavailable(
                stage="root_process",
                cause=psutil.AccessDenied(pid=101),
            )
        return 99.0

    monitor = ProcessTreeRSSMonitor(
        poll_interval_seconds=0.001,
        sampler=sampler,
    )
    monitor.start()
    assert second_call.wait(timeout=2.0)
    _wait_for_monitor_thread_to_stop(monitor)

    with pytest.raises(unavailable, match="root_process"):
        monitor.checkpoint()
    with pytest.raises(unavailable, match="root_process"):
        monitor.stop(check_limit=False)
    assert calls == [1, 2]
    assert monitor.current_rss_mb == 10.0
    assert monitor.peak_rss_mb == 10.0
    assert monitor.is_running is False


def test_invalid_background_sample_becomes_latched_unavailable_state():
    unavailable = _measurement_unavailable_type()
    calls = []

    def sampler():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return 10.0
        if len(calls) == 2:
            return object()
        return 99.0

    monitor = ProcessTreeRSSMonitor(
        poll_interval_seconds=0.001,
        sampler=sampler,
    )
    monitor.start()
    _wait_for_monitor_thread_to_stop(monitor)

    with pytest.raises(unavailable) as caught:
        monitor.checkpoint()
    assert caught.value.stage == "sampler_contract"
    assert caught.value.cause_type == "TypeError"
    with pytest.raises(unavailable, match="sampler_contract"):
        monitor.stop(check_limit=False)
    assert calls == [1, 2]
from src.utils.timing import (
    ProcessTreeRSSMonitor,
    ResourceLimitExceeded,
    TimingReport,
)


@pytest.mark.parametrize(
    "ceiling,poll",
    [(True, 0.01), ("10", 0.01), (0, 0.01), (10, True), (10, 0), (10, "0.1")],
)
def test_monitor_requires_exact_positive_numeric_configuration(ceiling, poll):
    with pytest.raises((TypeError, ValueError), match="ceiling|poll"):
        ProcessTreeRSSMonitor(rss_ceiling_mb=ceiling, poll_interval_seconds=poll)


def test_injected_sampler_tracks_peak_endpoint_and_aborts_cleanly():
    samples = iter([10.0, 12.0, 25.0, 8.0])
    monitor = ProcessTreeRSSMonitor(
        rss_ceiling_mb=20.0,
        poll_interval_seconds=60.0,
        sampler=lambda: next(samples),
    )
    monitor.start()
    assert monitor.current_rss_mb == 10.0
    assert monitor.checkpoint() == 12.0
    with pytest.raises(ResourceLimitExceeded) as caught:
        monitor.checkpoint()
    assert caught.value.observed_peak_mb == 25.0
    assert caught.value.ceiling_mb == 20.0
    monitor.stop(check_limit=False)
    assert monitor.current_rss_mb == 8.0
    assert monitor.peak_rss_mb == 25.0
    assert monitor.limit_exceeded is True
    assert monitor.is_running is False


def test_timing_report_keeps_peak_distinct_from_endpoint_and_aliases_peak():
    samples = iter([10.0, 25.0, 8.0])
    monitor = ProcessTreeRSSMonitor(
        poll_interval_seconds=60.0,
        sampler=lambda: next(samples),
    )
    report = TimingReport("resource_profile.py", rss_monitor=monitor)
    report.start()
    monitor.checkpoint()
    report.stop()

    payload = report.to_dict()
    assert payload["process_tree_peak_rss_mb"] == 25.0
    assert payload["peak_memory_mb"] == 25.0
    assert payload["endpoint_process_tree_rss_mb"] == 8.0
    assert payload["process_tree_peak_rss_mb"] != payload[
        "endpoint_process_tree_rss_mb"
    ]


def test_real_child_process_contributes_to_peak_but_not_endpoint():
    monitor = ProcessTreeRSSMonitor(poll_interval_seconds=0.005)
    monitor.start()
    baseline = monitor.current_rss_mb
    child = subprocess.Popen([
        sys.executable,
        "-c",
        "import time; payload=bytearray(24*1024*1024); time.sleep(0.35)",
    ])
    try:
        deadline = time.monotonic() + 2.0
        while child.poll() is None and monitor.peak_rss_mb < baseline + 12.0:
            time.sleep(0.01)
        child.wait(timeout=2.0)
        time.sleep(0.03)
        endpoint = monitor.checkpoint()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        monitor.stop(check_limit=False)
    assert monitor.peak_rss_mb >= baseline + 12.0
    assert endpoint < monitor.peak_rss_mb


def _fixture():
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    ids = ("q", "b", "a", "c", "d")
    candidates = {"q": ("d", "c", "b", "a")}
    return matrix, ids, candidates


def _assert_rankings_tolerance_equal(left, right):
    assert left.keys() == right.keys()
    for query_id in left:
        assert [item for item, _ in left[query_id]] == [
            item for item, _ in right[query_id]
        ]
        np.testing.assert_allclose(
            [score for _, score in left[query_id]],
            [score for _, score in right[query_id]],
            rtol=0.0,
            atol=1e-12,
        )


def test_batched_and_unbatched_top_k_agree_with_canonical_ties_and_candidates():
    matrix, ids, candidates = _fixture()
    observed_blocks = []
    batched = blockwise_cosine_top_k(
        matrix,
        ids,
        query_ids=("q",),
        candidates_by_query=candidates,
        k=3,
        query_block_size=1,
        candidate_block_size=2,
        block_observer=observed_blocks.append,
    )
    unbatched = blockwise_cosine_top_k(
        matrix,
        ids,
        query_ids=("q",),
        candidates_by_query=candidates,
        k=3,
        query_block_size=5,
        candidate_block_size=5,
    )
    _assert_rankings_tolerance_equal(batched, unbatched)
    assert [track_id for track_id, _ in batched["q"]] == ["a", "b", "c"]
    assert all(rows <= 1 and columns <= 2 for rows, columns in observed_blocks)
    assert (len(ids), len(ids)) not in observed_blocks


def test_batched_scores_tolerate_blas_shape_roundoff():
    rng = np.random.default_rng(804)
    matrix = rng.normal(size=(40, 9)).astype(np.float64)
    ids = tuple(f"track_{index:02d}" for index in range(40))
    candidates = {ids[0]: ids[1:]}
    batched = blockwise_cosine_top_k(
        matrix,
        ids,
        query_ids=(ids[0],),
        candidates_by_query=candidates,
        k=20,
        candidate_block_size=7,
    )
    unbatched = blockwise_cosine_top_k(
        matrix,
        ids,
        query_ids=(ids[0],),
        candidates_by_query=candidates,
        k=20,
        candidate_block_size=40,
    )

    _assert_rankings_tolerance_equal(batched, unbatched)


def test_short_unknown_duplicate_and_nonfinite_inputs_fail():
    matrix, ids, candidates = _fixture()
    with pytest.raises(SimilarityComputationError, match="fewer than k"):
        blockwise_cosine_top_k(
            matrix, ids, query_ids=("q",),
            candidates_by_query={"q": ("a", "b")}, k=3,
        )
    with pytest.raises(SimilarityComputationError, match="unknown"):
        blockwise_cosine_top_k(
            matrix, ids, query_ids=("q",),
            candidates_by_query={"q": ("a", "missing", "b")}, k=3,
        )
    with pytest.raises(SimilarityComputationError, match="duplicate"):
        blockwise_cosine_top_k(
            matrix, ids, query_ids=("q",),
            candidates_by_query={"q": ("a", "a", "b")}, k=3,
        )
    broken = matrix.copy()
    broken[0, 0] = np.nan
    with pytest.raises(SimilarityComputationError, match="finite"):
        blockwise_cosine_top_k(
            broken, ids, query_ids=("q",), candidates_by_query=candidates, k=3,
        )


def test_float32_uses_common_float64_source_and_enforces_tolerance():
    matrix, ids, candidates = _fixture()
    float64 = blockwise_cosine_top_k(
        matrix, ids, query_ids=("q",), candidates_by_query=candidates, k=3,
    )
    float32 = blockwise_cosine_top_k(
        matrix, ids, query_ids=("q",), candidates_by_query=candidates, k=3,
        use_float32=True, float32_atol=1e-6, candidate_block_size=2,
    )
    assert [item for item, _ in float32["q"]] == [item for item, _ in float64["q"]]
    np.testing.assert_allclose(
        [score for _, score in float32["q"]],
        [score for _, score in float64["q"]],
        rtol=0.0,
        atol=1e-6,
    )
    with pytest.raises(SimilarityComputationError, match="float32 tolerance"):
        blockwise_cosine_top_k(
            matrix, ids, query_ids=("q",), candidates_by_query=candidates, k=3,
            use_float32=True, float32_atol=0.0,
        )
    with pytest.raises(SimilarityComputationError, match="float64 source"):
        blockwise_cosine_top_k(
            matrix.astype(np.float32), ids, query_ids=("q",),
            candidates_by_query=candidates, k=3, use_float32=True,
        )


def test_breached_rss_ceiling_aborts_without_partial_ranking():
    matrix, ids, candidates = _fixture()
    samples = iter([10.0, 10.0, 50.0, 8.0])
    monitor = ProcessTreeRSSMonitor(
        rss_ceiling_mb=20.0,
        poll_interval_seconds=60.0,
        sampler=lambda: next(samples),
    )
    with pytest.raises(ResourceLimitExceeded):
        blockwise_cosine_top_k(
            matrix, ids, query_ids=("q",), candidates_by_query=candidates, k=3,
            candidate_block_size=2, rss_monitor=monitor,
        )
    assert monitor.is_running is False
