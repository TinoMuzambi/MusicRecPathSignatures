"""MR-03 contracts for level-zero-aware, fail-closed path signatures."""

import numpy as np
import pytest

import src.signatures.path_signatures as signature_module


def _failure_type():
    return getattr(signature_module, "TrackProcessingError")


def _valid_path(point_count=4):
    time = np.linspace(0.0, 1.0, point_count)
    features = np.vstack(
        [np.linspace(index, index + 0.5, point_count) for index in range(37)]
    ).T
    return np.column_stack([time, features]).astype(np.float32)


@pytest.mark.parametrize(
    ("order", "length"), [(1, 24), (2, 553), (3, 12720)]
)
def test_signature_length_includes_level_zero(order, length):
    assert signature_module.PathSignature.get_signature_length_for_order(order, 23) == length


def test_real_esig_includes_level_zero_and_expected_dimension():
    path = _valid_path(point_count=4)
    signature = signature_module.PathSignature(
        order=2, normalise_signatures=False
    ).compute_signature(path, track_id="real")
    assert signature.shape == (1483,)
    assert signature[0] == pytest.approx(1.0)


def test_complete_level_zero_signature_is_l2_normalised(monkeypatch):
    raw = np.zeros(1483, dtype=float)
    raw[0] = 1.0
    raw[1:5] = [0.1, -0.2, 0.3, -0.4]
    monkeypatch.setattr(signature_module.esig, "stream2sig", lambda *_: raw.copy())
    signature = signature_module.PathSignature(order=2).compute_signature(
        _valid_path(), track_id="normalised"
    )
    expected_divisor = np.sqrt(1.0 + np.sum(np.square(raw[1:])))
    assert np.linalg.norm(raw) == expected_divisor
    np.testing.assert_array_equal(signature, raw / expected_divisor)
    assert np.linalg.norm(signature) == pytest.approx(1.0)
    assert signature[0] == 1.0 / expected_divisor


def test_real_complete_signatures_feed_ordinary_cosine_adapter():
    from src.recommendation.path_signature_cosine import (
        rank_path_signature_candidates,
    )

    first_path = _valid_path(point_count=4)
    second_path = first_path.copy()
    second_path[:, 1:] = np.square(second_path[:, 1:] / 25.0)
    model = signature_module.PathSignature(order=2)
    first = model.compute_signature(first_path, track_id="first")
    second = model.compute_signature(second_path, track_id="second")

    ranked = rank_path_signature_candidates(
        query_track_id="first",
        candidate_ids=["second"],
        excluded_ids=[],
        signatures={"first": first, "second": second},
        top_k=1,
    )
    expected = float(np.dot(first, second))
    assert first[0] > 0.0 and second[0] > 0.0
    assert ranked == (("second", pytest.approx(expected)),)
    assert expected == pytest.approx(
        np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )


def test_level_zero_scalar_must_be_present_before_normalisation(monkeypatch):
    malformed = np.arange(1483, dtype=float)
    monkeypatch.setattr(
        signature_module.esig, "stream2sig", lambda *_: malformed.copy()
    )
    with pytest.raises(_failure_type()) as caught:
        signature_module.PathSignature(order=2).compute_signature(
            _valid_path(), track_id="missing-level-zero"
        )
    assert caught.value.reason_code == "signature_level_zero"


@pytest.mark.parametrize(
    ("replacement", "reason_code"),
    [
        (RuntimeError("boom"), "esig_exception"),
        (np.zeros(1482), "signature_length"),
        (np.full(1483, np.nan), "signature_non_finite"),
    ],
)
def test_esig_failure_never_becomes_a_retained_fallback(
    monkeypatch, replacement, reason_code
):
    if isinstance(replacement, Exception):
        def fail(*_args, **_kwargs):
            raise replacement

        monkeypatch.setattr(signature_module.esig, "stream2sig", fail)
    else:
        monkeypatch.setattr(
            signature_module.esig, "stream2sig", lambda *_: replacement.copy()
        )

    with pytest.raises(_failure_type()) as caught:
        signature_module.PathSignature(order=2).compute_signature(
            _valid_path(), track_id="bad"
        )

    assert caught.value.track_id == "bad"
    assert caught.value.stage == "signature"
    assert caught.value.reason_code == reason_code


def test_short_path_is_rejected_before_esig(monkeypatch):
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return np.zeros(1483)

    monkeypatch.setattr(signature_module.esig, "stream2sig", should_not_run)
    with pytest.raises(_failure_type()) as caught:
        signature_module.PathSignature(order=2).compute_signature(
            _valid_path(point_count=2), track_id="short"
        )
    assert caught.value.reason_code == "path_too_short"
    assert called is False


def test_signature_batch_separates_accepted_tracks_and_failures(monkeypatch):
    monkeypatch.setattr(
        signature_module.esig,
        "stream2sig",
        lambda *_: np.arange(1.0, 1484.0),
    )
    result = signature_module.PathSignature(order=2).compute_signatures_dict(
        {
            "ok": {"multi_dimensional_series": _valid_path()},
            "bad": {"multi_dimensional_series": np.full((4, 38), np.nan)},
            "missing": {},
        }
    )
    assert tuple(result.accepted) == ("ok",)
    assert {row["track_id"] for row in result.failures} == {"bad", "missing"}
    assert not np.allclose(result.accepted["ok"], 0.0)


def test_signature_batch_rejects_whole_normalised_id_collision(monkeypatch):
    monkeypatch.setattr(
        signature_module.esig,
        "stream2sig",
        lambda *_: np.concatenate(([1.0], np.full(1482, 0.01))),
    )
    result = signature_module.PathSignature(order=2).compute_signatures_dict(
        {
            "7": {"multi_dimensional_series": _valid_path()},
            "007": {"multi_dimensional_series": _valid_path()},
            "ok": {"multi_dimensional_series": _valid_path()},
        }
    )

    assert tuple(result.accepted) == ("ok",)
    assert len(result.failures) == 1
    assert result.failures[0]["track_id"] == "7"
    assert result.failures[0]["reason_code"] == "duplicate_track"


def test_signature_duplicate_outcome_is_invariant_to_batch_size(monkeypatch):
    monkeypatch.setattr(
        signature_module.esig,
        "stream2sig",
        lambda *_: np.concatenate(([1.0], np.full(1482, 0.01))),
    )
    features = {
        "7": {"multi_dimensional_series": _valid_path()},
        "007": {"multi_dimensional_series": _valid_path()},
        "ok": {"multi_dimensional_series": _valid_path()},
    }
    model = signature_module.PathSignature(order=2)

    one = model.compute_signatures_batch(features, batch_size=1)
    two = model.compute_signatures_batch(features, batch_size=2)
    ten = model.compute_signatures_batch(features, batch_size=10)

    assert tuple(one.accepted) == tuple(two.accepted) == tuple(ten.accepted) == ("ok",)
    assert one.failures == two.failures == ten.failures
