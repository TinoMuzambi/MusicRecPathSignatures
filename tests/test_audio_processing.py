"""MR-03 contracts for fail-closed audio and 38-channel path processing."""

import numpy as np
import pytest
import soundfile as sf

import src.audio.processing as processing


def _failure_type():
    return getattr(processing, "TrackProcessingError")


def _path_inputs(frame_count=5):
    pitch = np.linspace(100.0, 110.0, frame_count)
    loudness = np.linspace(0.1, 0.5, frame_count)
    mfccs = np.vstack(
        [np.linspace(index, index + 1.0, frame_count) for index in range(20)]
    )
    audio = np.linspace(-0.5, 0.5, (frame_count - 1) * processing.HOP_LENGTH)
    return pitch, loudness, mfccs, audio


def _extra_features(frame_count=5):
    chroma = np.vstack(
        [np.linspace(index / 10.0, index / 10.0 + 0.1, frame_count) for index in range(12)]
    )
    spectral_centroid = np.linspace(1.0, 2.0, frame_count)
    spectral_bandwidth = np.linspace(0.5, 0.9, frame_count)
    zero_crossing_rate = np.linspace(0.0, 0.3, frame_count)
    return {
        "chroma": chroma,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "zero_crossing_rate": zero_crossing_rate,
    }


def test_load_audio_rejects_corrupt_and_silent_files(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    silent = tmp_path / "silent.wav"
    corrupt.write_bytes(b"not audio" * 200)
    sf.write(silent, np.zeros(22050), 22050)

    with pytest.raises(_failure_type()) as corrupt_error:
        processing.load_audio(str(corrupt), track_id="corrupt")
    with pytest.raises(_failure_type()) as silent_error:
        processing.load_audio(str(silent), track_id="silent")

    assert corrupt_error.value.reason_code in {"invalid_audio", "unreadable_audio"}
    assert silent_error.value.reason_code == "silent_audio"
    assert corrupt_error.value.stage == silent_error.value.stage == "audio_load"


def test_load_audio_distinguishes_an_unusable_path_argument():
    with pytest.raises(_failure_type()) as caught:
        processing.load_audio(object(), track_id="bad-path")

    assert caught.value.track_id == "bad-path"
    assert caught.value.stage == "audio_load"
    assert caught.value.reason_code == "invalid_audio_path"


def test_create_path_has_frozen_channel_order_time_and_finite_clipping():
    pitch, loudness, mfccs, audio = _path_inputs()
    mfccs[0] = 100.0
    path = processing.create_multidimensional_timeseries(
        pitch, loudness, _extra_features(), mfccs, 22050, audio, track_id="track-1"
    )

    assert path.shape == (pitch.size, 38)
    assert path.dtype == np.float32
    assert tuple(processing.SIGNATURE_CHANNELS) == (
        "time",
        "pitch",
        "loudness",
        *(f"mfcc_{index:02d}" for index in range(1, 21)),
        *(f"chroma_{index:02d}" for index in range(1, 13)),
        "spectral_centroid",
        "spectral_bandwidth",
        "zero_crossing_rate",
    )
    assert path[0, 0] == 0.0
    assert path[-1, 0] == 1.0
    assert np.all(np.diff(path[:, 0]) >= 0)
    assert np.isfinite(path).all()
    assert np.max(path[:, 1:]) <= 5.0
    assert np.min(path[:, 1:]) >= -5.0


def test_create_path_columns_match_declared_pitch_loudness_and_mfcc_order():
    frame_count = 4
    sample_rate = 22050
    audio = np.linspace(
        -0.5, 0.5, (frame_count - 1) * processing.HOP_LENGTH
    )
    pitch = np.linspace(0.1, 0.4, frame_count)
    loudness = np.linspace(-0.4, -0.1, frame_count)
    mfccs = np.vstack(
        [np.linspace(-4.0 + index / 10.0, -3.7 + index / 10.0, frame_count)
         for index in range(20)]
    )
    extra = _extra_features(frame_count)

    path = processing.create_multidimensional_timeseries(
        pitch, loudness, extra, mfccs, sample_rate, audio, track_id="ordered"
    )
    def standardised(values):
        values = np.asarray(values, dtype=float)
        return (values - np.mean(values)) / np.std(values)

    np.testing.assert_allclose(path[:, 1], standardised(pitch))
    np.testing.assert_allclose(path[:, 2], standardised(loudness))
    for index in range(20):
        np.testing.assert_allclose(
            path[:, 3 + index],
            standardised(mfccs[index]),
        )
    for index in range(12):
        np.testing.assert_allclose(
            path[:, 23 + index],
            standardised(extra["chroma"][index]),
        )
    np.testing.assert_allclose(
        path[:, 35],
        standardised(extra["spectral_centroid"]),
    )
    np.testing.assert_allclose(
        path[:, 36],
        standardised(extra["spectral_bandwidth"]),
    )
    np.testing.assert_allclose(
        path[:, 37],
        standardised(extra["zero_crossing_rate"]),
    )


@pytest.mark.parametrize("field", ["pitch", "loudness", "mfccs", "audio"])
def test_create_path_rejects_non_finite_input_before_repair(field):
    pitch, loudness, mfccs, audio = _path_inputs()
    values = {"pitch": pitch, "loudness": loudness, "mfccs": mfccs, "audio": audio}
    values[field].flat[0] = np.nan

    with pytest.raises(_failure_type()) as caught:
        processing.create_multidimensional_timeseries(
            values["pitch"],
            values["loudness"],
            _extra_features(),
            values["mfccs"],
            22050,
            values["audio"],
            track_id=" 009 ",
        )

    assert caught.value.track_id == "9"
    assert caught.value.reason_code == "non_finite_feature"


def test_create_path_rejects_missing_extra_features():
    pitch, loudness, mfccs, audio = _path_inputs()

    with pytest.raises(_failure_type()) as caught:
        processing.create_multidimensional_timeseries(
            pitch, loudness, None, mfccs, 22050, audio, track_id="track-1"
        )

    assert caught.value.reason_code == "invalid_feature"

    extra = _extra_features()
    del extra["chroma"]
    with pytest.raises(_failure_type()) as caught_missing:
        processing.create_multidimensional_timeseries(
            pitch, loudness, extra, mfccs, 22050, audio, track_id="track-1"
        )

    assert caught_missing.value.reason_code == "missing_feature"


@pytest.mark.parametrize(
    ("path", "reason_code"),
    [
        (np.ones((38, 30)), "path_orientation"),
        (np.ones((30, 37)), "path_channels"),
        (np.ones((2, 38)), "path_too_short"),
        (np.full((30, 38), np.nan), "path_non_finite"),
    ],
)
def test_validate_signature_path_rejects_malformed_paths(path, reason_code):
    validate = getattr(processing, "validate_signature_path")
    with pytest.raises(_failure_type()) as caught:
        validate(path, track_id="track", order=2)
    assert caught.value.reason_code == reason_code
    assert caught.value.track_id == "track"
    assert caught.value.stage == "signature_path"


def test_validate_signature_path_accepts_json_lists_without_reshaping():
    validate = getattr(processing, "validate_signature_path")
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 4), np.ones((4, 37), dtype=float)]
    )
    validated = validate(path.tolist(), track_id="track", order=2)
    assert validated.shape == (4, 38)
    assert validated.dtype == np.float32
