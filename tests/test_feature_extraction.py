"""MR-03 contracts for canonical audio-feature extraction."""

import json

import numpy as np
import pytest
import soundfile as sf

import src.audio.feature_extraction as feature_module


def _valid_features(frame_count=4):
    return {
        "mfccs": np.arange(20 * frame_count, dtype=float).reshape(20, frame_count),
        "chroma": np.arange(12 * frame_count, dtype=float).reshape(12, frame_count),
        "spectral_centroid": np.linspace(1.0, 2.0, frame_count),
        "spectral_bandwidth": np.linspace(2.0, 3.0, frame_count),
        "zero_crossing_rate": np.linspace(0.1, 0.4, frame_count),
        "loudness": np.linspace(0.2, 0.8, frame_count),
    }


def _failure_type():
    return getattr(feature_module, "TrackProcessingError")


def test_traditional_vector_is_representation_invariant_and_axis_correct():
    build = getattr(feature_module, "build_traditional_feature_vector")
    arrays = _valid_features()
    json_lists = {key: value.tolist() for key, value in arrays.items()}

    array_vector = build(" 007 ", arrays)
    list_vector = build("007", json_lists)

    np.testing.assert_allclose(array_vector, list_vector)
    assert array_vector.shape == (72,)
    # MFCC channel 0 is [0, 1, 2, 3]: channel mean then channel std.
    assert array_vector[0] == pytest.approx(1.5)
    assert array_vector[20] == pytest.approx(np.std([0.0, 1.0, 2.0, 3.0]))
    # Chroma starts after 20 MFCC means and 20 MFCC standard deviations.
    assert array_vector[40] == pytest.approx(1.5)
    assert np.isfinite(array_vector).all()


def test_traditional_sequence_fields_keep_declared_mean_std_order():
    build = getattr(feature_module, "build_traditional_feature_vector")
    features = _valid_features()
    distinct_values = {
        "spectral_centroid": 11.0,
        "spectral_bandwidth": 22.0,
        "zero_crossing_rate": 33.0,
        "loudness": 44.0,
    }
    for field, value in distinct_values.items():
        features[field] = np.full(4, value)

    vector = build("sequence-order", features)

    np.testing.assert_array_equal(
        vector[64:72],
        np.array([11.0, 0.0, 22.0, 0.0, 33.0, 0.0, 44.0, 0.0]),
    )


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (lambda values: values.pop("loudness"), "missing_feature"),
        (
            lambda values: values.__setitem__("mfccs", values["mfccs"].T),
            "feature_orientation",
        ),
        (
            lambda values: values.__setitem__(
                "spectral_centroid", np.arange(3, dtype=float)
            ),
            "inconsistent_frames",
        ),
        (
            lambda values: values["mfccs"].__setitem__((0, 0), np.nan),
            "non_finite_feature",
        ),
        (
            lambda values: values.__setitem__("chroma", np.empty((12, 0))),
            "empty_feature",
        ),
    ],
)
def test_traditional_vector_rejects_invalid_schema_with_track_context(
    mutate, reason_code
):
    build = getattr(feature_module, "build_traditional_feature_vector")
    features = _valid_features()
    mutate(features)

    with pytest.raises(_failure_type()) as caught:
        build(" 007 ", features)

    assert caught.value.track_id == "7"
    assert caught.value.stage == "traditional_features"
    assert caught.value.reason_code == reason_code
    assert caught.value.reason


@pytest.mark.parametrize(
    "value",
    [
        np.full((20, 4), True, dtype=bool),
        np.full((20, 4), "1.0", dtype=str),
    ],
)
def test_traditional_vector_rejects_non_numeric_source_dtypes(value):
    build = getattr(feature_module, "build_traditional_feature_vector")
    features = _valid_features()
    features["mfccs"] = value

    with pytest.raises(_failure_type()) as caught:
        build("weak-type", features)

    assert caught.value.reason_code == "invalid_feature"
    assert caught.value.stage == "traditional_features"


def test_valid_audio_extraction_exposes_loudness_and_ordered_path(tmp_path):
    sample_rate = 22050
    times = np.arange(sample_rate, dtype=float) / sample_rate
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, np.sin(2 * np.pi * 440 * times), sample_rate)

    extractor = feature_module.AudioFeatureExtractor(target_length=32)
    features = extractor.extract_features(str(audio_path), track_id="tone")

    assert set(_valid_features()).issubset(features)
    assert features["mfccs"].shape == (20, 32)
    assert features["chroma"].shape == (12, 32)
    assert features["loudness"].shape == (32,)
    assert features["multi_dimensional_series"].shape[1] == 38
    assert np.isfinite(features["multi_dimensional_series"]).all()


def test_batch_outcomes_are_deterministic_and_disjoint(tmp_path):
    sample_rate = 22050
    valid_path = tmp_path / "valid.wav"
    silent_path = tmp_path / "silent.wav"
    corrupt_path = tmp_path / "corrupt.wav"
    output_one = tmp_path / "out-one"
    output_two = tmp_path / "out-two"
    times = np.arange(sample_rate, dtype=float) / sample_rate
    sf.write(valid_path, np.sin(2 * np.pi * 220 * times), sample_rate)
    sf.write(silent_path, np.zeros(sample_rate), sample_rate)
    corrupt_path.write_bytes(b"not an audio file" * 100)

    extractor = feature_module.AudioFeatureExtractor(target_length=16)
    paths = [str(silent_path), str(valid_path), str(corrupt_path)]
    first = extractor.extract_features_batch(paths, output_dir=output_one)
    second = extractor.extract_features_batch(paths, output_dir=output_two)

    assert tuple(first.accepted) == (str(valid_path),)
    assert {record["track_id"] for record in first.failures} == {
        str(silent_path),
        str(corrupt_path),
    }
    assert (output_one / "accepted_tracks.json").read_bytes() == (
        output_two / "accepted_tracks.json"
    ).read_bytes()
    assert (output_one / "track_failures.jsonl").read_bytes() == (
        output_two / "track_failures.jsonl"
    ).read_bytes()

    accepted_payload = json.loads(
        (output_one / "accepted_tracks.json").read_text(encoding="utf-8")
    )
    failure_rows = [
        json.loads(line)
        for line in (output_one / "track_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    accepted_ids = set(accepted_payload["ordered_track_ids"])
    failed_ids = {row["track_id"] for row in failure_rows}
    assert accepted_ids.isdisjoint(failed_ids)
    assert accepted_ids == {str(valid_path)}
    assert all("timestamp" not in row and "exception" not in row for row in failure_rows)
    assert [row["track_id"] for row in failure_rows] == sorted(failed_ids)


def test_batch_rejects_every_member_of_a_normalised_id_collision(monkeypatch):
    extractor = feature_module.AudioFeatureExtractor(target_length=4)
    calls = []

    def should_not_extract(*args, **kwargs):
        calls.append((args, kwargs))
        return _valid_features()

    monkeypatch.setattr(extractor, "extract_features", should_not_extract)

    result = extractor.extract_features_batch(["7", "007"])

    assert result.accepted == {}
    assert calls == []
    assert len(result.failures) == 1
    assert result.failures[0]["track_id"] == "7"
    assert result.failures[0]["reason_code"] == "duplicate_track"


def test_batch_reraises_unexpected_programming_errors(monkeypatch):
    extractor = feature_module.AudioFeatureExtractor(target_length=4)

    def broken_extractor(*_args, **_kwargs):
        raise AttributeError("programming defect")

    monkeypatch.setattr(extractor, "extract_features", broken_extractor)

    with pytest.raises(AttributeError, match="programming defect"):
        extractor.extract_features_batch(["track"])
