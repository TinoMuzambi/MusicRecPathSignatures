"""V30 contracts for the compact feature, population and EDA boundary."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.audio.feature_extraction as feature_extraction
from src.data.synthetic_users import (
    CANONICAL_ARCHETYPE_DISTRIBUTION,
    SyntheticUserGenerator,
    build_synthetic_population,
)


def _feature_bundle_module():
    module_path = Path(__file__).parents[1] / "src" / "utils" / "feature_bundle.py"
    assert module_path.is_file(), "compact feature-bundle module must be implemented"
    return importlib.import_module("src.utils.feature_bundle")


def _compact_feature(seed: int, frames: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "multi_dimensional_series": rng.normal(size=(frames, 38)),
        "traditional_feature_vector": rng.normal(size=72),
    }


def _tracks(count: int = 10) -> list[dict]:
    return [
        {
            "track_id": index + 1,
            "title": f"Track {index + 1}",
            "artist": f"Artist {index % 3}",
            "genre": "Rock" if index < count // 2 else "Jazz",
            "duration": 120.0 + index,
        }
        for index in range(count)
    ]


def _canonical_tracks(count: int = 1000) -> list[dict]:
    genres = ("Rock", "Pop", "Electronic", "Jazz", "Classical", "Hip-Hop")
    return [
        {
            "track_id": index + 1,
            "title": f"Track {index + 1}",
            "artist": f"Artist {index % 37}",
            "genre": genres[index % len(genres)],
            "duration": 120.0 + index,
        }
        for index in range(count)
    ]


def _resign_population(population: dict) -> None:
    from src.data import synthetic_users

    population["diagnostics"] = synthetic_users._population_diagnostics(
        users=population["users"],
        interactions=population["interactions"],
        splits=population["splits"],
        track_metadata=population["configuration"]["track_metadata"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_exact_traditional_feature_names_and_compact_vector_validation():
    TRADITIONAL_FEATURE_NAMES = getattr(
        feature_extraction, "TRADITIONAL_FEATURE_NAMES", None
    )
    assert TRADITIONAL_FEATURE_NAMES is not None
    build_traditional_feature_vector = feature_extraction.build_traditional_feature_vector
    assert len(TRADITIONAL_FEATURE_NAMES) == 72
    assert len(set(TRADITIONAL_FEATURE_NAMES)) == 72
    assert TRADITIONAL_FEATURE_NAMES[:3] == (
        "mfcc_01_mean",
        "mfcc_02_mean",
        "mfcc_03_mean",
    )
    assert TRADITIONAL_FEATURE_NAMES[-8:] == (
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "loudness_mean",
        "loudness_std",
    )

    vector = np.arange(72, dtype=np.float64)
    np.testing.assert_array_equal(
        build_traditional_feature_vector(
            "7", {"traditional_feature_vector": vector.tolist()}
        ),
        vector,
    )
    with pytest.raises(Exception, match="72 finite values"):
        build_traditional_feature_vector(
            "7", {"traditional_feature_vector": vector[:-1]}
        )


def test_compact_extraction_admits_the_exact_stored_dtypes(monkeypatch):
    extractor = feature_extraction.AudioFeatureExtractor()
    expanded = _compact_feature(7, 3)
    expanded["discarded_sequence"] = np.arange(3, dtype=np.float64)
    monkeypatch.setattr(
        extractor,
        "extract_features",
        lambda _path, *, track_id: expanded,
    )
    compact = extractor.extract_compact_features("unused.wav", track_id="7")
    assert set(compact) == {
        "multi_dimensional_series",
        "traditional_feature_vector",
    }
    assert compact["multi_dimensional_series"].dtype.str == "<f4"
    assert compact["traditional_feature_vector"].dtype.str == "<f8"
    assert compact["multi_dimensional_series"].flags.c_contiguous

    expanded["multi_dimensional_series"][0, 0] = np.finfo(np.float64).max
    with pytest.raises(Exception, match="float32|finite"):
        extractor.extract_compact_features("unused.wav", track_id="7")


def test_feature_bundle_is_little_endian_compact_strict_and_hash_bound(tmp_path):
    bundle_module = _feature_bundle_module()
    write_feature_bundle = bundle_module.write_feature_bundle
    load_feature_bundle = bundle_module.load_feature_bundle
    FeatureBundleError = bundle_module.FeatureBundleError
    TRADITIONAL_FEATURE_NAMES = feature_extraction.TRADITIONAL_FEATURE_NAMES
    output = tmp_path / "bundle"
    source = {
        "2": _compact_feature(2, 3),
        "10": _compact_feature(10, 5),
    }

    manifest = write_feature_bundle(source, output)

    assert manifest["schema_version"] == 1
    assert manifest["ordered_track_ids"] == ["10", "2"]
    assert manifest["path_channel_count"] == 38
    assert manifest["traditional_feature_names"] == list(TRADITIONAL_FEATURE_NAMES)
    assert manifest["files"]["path_values.f32le"]["dtype"] == "<f4"
    assert manifest["files"]["path_offsets.i64le"]["dtype"] == "<i8"
    assert manifest["files"]["traditional_features.f64le"]["dtype"] == "<f8"
    assert (output / "path_values.f32le").stat().st_size == (5 + 3) * 38 * 4
    assert np.fromfile(output / "path_offsets.i64le", dtype="<i8").tolist() == [
        0,
        5 * 38,
        8 * 38,
    ]
    for filename in (
        "track_ids.json",
        "path_values.f32le",
        "path_offsets.i64le",
        "traditional_features.f64le",
        "bundle_manifest.json",
    ):
        sidecar = (output / f"{filename}.sha256").read_text(encoding="ascii")
        assert sidecar == f"{_sha256(output / filename)}  {filename}\n"

    bundle = load_feature_bundle(
        output, expected_track_ids=("10", "2"), expected_path_channels=38
    )
    assert tuple(bundle) == ("10", "2")
    np.testing.assert_allclose(
        bundle["10"]["multi_dimensional_series"],
        source["10"]["multi_dimensional_series"].astype("<f4"),
    )
    np.testing.assert_array_equal(
        bundle["2"]["traditional_feature_vector"],
        source["2"]["traditional_feature_vector"],
    )

    with (output / "path_values.f32le").open("r+b") as target:
        target.seek(0)
        original = target.read(1)
        target.seek(0)
        target.write(bytes([original[0] ^ 1]))
    with pytest.raises(FeatureBundleError, match="SHA-256"):
        load_feature_bundle(output)


def test_feature_bundle_refuses_existing_output_and_non_finite_data(tmp_path):
    bundle_module = _feature_bundle_module()
    write_feature_bundle = bundle_module.write_feature_bundle
    FeatureBundleError = bundle_module.FeatureBundleError
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        write_feature_bundle({"1": _compact_feature(1, 2)}, existing)

    invalid = _compact_feature(1, 2)
    invalid["traditional_feature_vector"][0] = np.nan
    with pytest.raises(FeatureBundleError, match="finite"):
        write_feature_bundle({"1": invalid}, tmp_path / "invalid")


@pytest.mark.parametrize("mutation", ("manifest", "entry", "dtype"))
def test_feature_bundle_rejects_resigned_schema_mutations(tmp_path, mutation):
    bundle_module = _feature_bundle_module()
    output = tmp_path / mutation
    bundle_module.write_feature_bundle({"1": _compact_feature(1, 2)}, output)
    manifest_path = output / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "manifest":
        manifest["unexpected"] = True
    elif mutation == "entry":
        manifest["files"]["path_values.f32le"]["unexpected"] = True
    else:
        manifest["files"]["path_values.f32le"]["dtype"] = ">f4"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = _sha256(manifest_path)
    (output / "bundle_manifest.json.sha256").write_text(
        f"{digest}  bundle_manifest.json\n", encoding="ascii"
    )
    with pytest.raises(bundle_module.FeatureBundleError, match="manifest|entry|dtype"):
        bundle_module.load_feature_bundle(output)


def test_feature_bundle_expected_ids_must_be_unique_sorted_and_exact(tmp_path):
    bundle_module = _feature_bundle_module()
    output = tmp_path / "bundle"
    bundle_module.write_feature_bundle(
        {"1": _compact_feature(1, 2), "2": _compact_feature(2, 2)}, output
    )
    with pytest.raises(bundle_module.FeatureBundleError, match="duplicate"):
        bundle_module.load_feature_bundle(output, expected_track_ids=("1", "1"))
    with pytest.raises(bundle_module.FeatureBundleError, match="sorted"):
        bundle_module.load_feature_bundle(output, expected_track_ids=("2", "1"))
    with pytest.raises(bundle_module.FeatureBundleError, match="match"):
        bundle_module.load_feature_bundle(output, expected_track_ids=("1", "3"))


def test_preferred_and_random_samples_are_disjoint_and_declared_count_is_realised():
    tracks = _tracks()
    generator = SyntheticUserGenerator(random_seed=3)
    stats = generator._calculate_track_statistics(tracks)
    profile = {
        "engagement_level": 1.0,
        "interaction_rate": 1.0,
        "preferred_genres": ["Rock"],
        "popularity_bias": 0.2,
        "diversity_preference": 0.2,
        "novelty_seeking": 0.2,
    }

    interactions = generator._generate_user_interactions(profile, tracks, stats)

    assert profile["declared_interaction_count"] == len(interactions)
    assert profile["realised_interaction_count"] == len(interactions)
    assert len(interactions) == len(set(interactions))


@pytest.mark.parametrize(
    "distribution",
    [
        {**CANONICAL_ARCHETYPE_DISTRIBUTION, "invented": 0.0},
        {**CANONICAL_ARCHETYPE_DISTRIBUTION, "music_enthusiast": float("nan")},
        {**CANONICAL_ARCHETYPE_DISTRIBUTION, "music_enthusiast": True},
    ],
)
def test_synthetic_population_rejects_invalid_archetype_contract(distribution):
    generator = SyntheticUserGenerator(2025)
    with pytest.raises(ValueError, match="archetype"):
        generator.generate_users(_tracks(), 2, distribution)


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", ""),
        ("artist", "Unknown"),
        ("genre", None),
        ("duration", float("nan")),
        ("duration", 0.0),
    ],
)
def test_synthetic_population_rejects_incomplete_or_unknown_metadata(field, value):
    tracks = _tracks()
    tracks[0][field] = value
    with pytest.raises((ValueError, TypeError), match=field):
        build_synthetic_population(tracks)


def test_population_validator_requires_declared_realised_and_split_consistency():
    validate_synthetic_population = getattr(
        importlib.import_module("src.data.synthetic_users"),
        "validate_synthetic_population",
    )
    population = build_synthetic_population(_canonical_tracks())
    validate_synthetic_population(population, expected_track_count=1000)
    user_id = sorted(population["users"])[0]
    population["users"][user_id]["realised_interaction_count"] += 1
    with pytest.raises(ValueError, match="declared|realised"):
        validate_synthetic_population(population, expected_track_count=1000)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("archetype", "Invented label", "archetype profile"),
        ("description", "Invented description", "archetype profile"),
        ("interaction_rate", 0.123, "archetype profile"),
        ("preferred_genres", ["Invented genre"], "preferred_genres"),
    ],
)
def test_population_validator_binds_each_user_to_the_canonical_archetype(
    field, value, error
):
    from src.data.synthetic_users import validate_synthetic_population

    population = build_synthetic_population(_canonical_tracks())
    user_id = next(
        user_id
        for user_id, user in sorted(population["users"].items())
        if user["archetype_key"] == "genre_specialist"
    )
    mutated = copy.deepcopy(population)
    mutated["users"][user_id][field] = value
    _resign_population(mutated)

    with pytest.raises(ValueError, match=error):
        validate_synthetic_population(mutated, expected_track_count=1000)


@pytest.mark.parametrize(
    "preferred_genres",
    [
        ["Invented genre"],
        ["Rock", "Pop", "Electronic", "Jazz"],
    ],
)
def test_population_validator_enforces_the_explorer_catalogue_rule(
    preferred_genres,
):
    from src.data.synthetic_users import validate_synthetic_population

    population = build_synthetic_population(_canonical_tracks())
    user_id = next(
        user_id
        for user_id, user in sorted(population["users"].items())
        if user["archetype_key"] == "explorer"
    )
    population["users"][user_id]["preferred_genres"] = preferred_genres
    _resign_population(population)

    with pytest.raises(ValueError, match="preferred_genres"):
        validate_synthetic_population(population, expected_track_count=1000)


def test_population_validator_requires_fixed_focus_genres_in_the_catalogue():
    with pytest.raises(ValueError, match="catalogue"):
        build_synthetic_population(_tracks(1000))


def test_population_validator_recomputes_the_exact_seeded_splits():
    from src.data.synthetic_users import validate_synthetic_population

    population = build_synthetic_population(_canonical_tracks())
    user_id = sorted(population["users"])[0]
    track_id, interaction = next(
        iter(population["splits"]["train"][user_id].items())
    )
    del population["splits"]["train"][user_id][track_id]
    population["splits"]["validation"][user_id][track_id] = interaction
    _resign_population(population)

    with pytest.raises(ValueError, match="seeded split"):
        validate_synthetic_population(population, expected_track_count=1000)


def test_population_validator_requires_every_canonical_archetype_to_be_realised():
    from src.data.synthetic_users import SyntheticUserGenerator
    from src.data.synthetic_users import validate_synthetic_population

    population = build_synthetic_population(_canonical_tracks())
    casual = SyntheticUserGenerator(2025).archetypes["casual_listener"]
    for user in population["users"].values():
        if user["archetype_key"] == "genre_specialist":
            user["archetype_key"] = "casual_listener"
            user["archetype"] = casual.name
            user["description"] = casual.description
            user["interaction_rate"] = casual.interaction_rate
            user["preferred_genres"] = list(casual.genre_focus)
    _resign_population(population)

    with pytest.raises(ValueError, match="archetype.*realised"):
        validate_synthetic_population(population, expected_track_count=1000)


def test_population_schema_and_json_loader_are_exact_and_fail_closed(tmp_path):
    from src.data.synthetic_users import validate_synthetic_population
    from src.scripts.generate_synthetic_users import load_population_file

    population = build_synthetic_population(_canonical_tracks())
    population["unexpected"] = True
    with pytest.raises(ValueError, match="fields|schema"):
        validate_synthetic_population(population, expected_track_count=1000)

    path = tmp_path / "population.json"
    malformed = b'{"schema_version":1,"schema_version":NaN}\n'
    path.write_bytes(malformed)
    digest = hashlib.sha256(malformed).hexdigest()
    (tmp_path / "population.json.sha256").write_text(
        f"{digest}  population.json\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="duplicate|non-finite"):
        load_population_file(path)


def test_population_is_one_hash_bound_reusable_file(tmp_path):
    from src.scripts.generate_synthetic_users import (
        export_population_file,
        load_population_file,
    )

    population = build_synthetic_population(_canonical_tracks())
    path = tmp_path / "canonical_synthetic_population.json"
    digest = export_population_file(population, path)
    assert (tmp_path / f"{path.name}.sha256").read_text(encoding="ascii") == (
        f"{digest}  {path.name}\n"
    )
    assert load_population_file(path, expected_track_count=1000) == population

    with path.open("r+b") as target:
        original = target.read(1)
        target.seek(0)
        target.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        load_population_file(path, expected_track_count=1000)


def _fma_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [
            ("set", "subset"),
            ("track", "genre_top"),
            ("track", "title"),
            ("track", "duration"),
            ("artist", "name"),
        ]
    )
    rows = [
        ["medium", "Rock", "Rock one", 120.0, "Artist A"],
        ["medium", "Rock", "Rock two", 121.0, "Artist B"],
        ["medium", "Jazz", "Jazz one", 122.0, "Artist C"],
        ["medium", "Jazz", "Jazz two", 123.0, "Artist D"],
    ]
    return pd.DataFrame(rows, index=[1, 2, 3, 4], columns=columns)


def test_strict_fma_metadata_and_genre_queues_are_exact_and_deterministic(tmp_path):
    from src.scripts.robust_track_processing import (
        build_genre_candidate_queues,
        validate_fma_medium_metadata,
    )

    audio_root = tmp_path / "audio"
    for track_id in range(1, 5):
        path = audio_root / "000" / f"{track_id:06d}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")
    records = validate_fma_medium_metadata(_fma_frame(), audio_root)
    first = build_genre_candidate_queues(records, target_count=3, seed=2025)
    second = build_genre_candidate_queues(records, target_count=3, seed=2025)

    assert first == second
    assert sum(item["quota"] for item in first.values()) == 3
    assert {track["track_id"] for item in first.values() for track in item["candidates"]} == {
        "1",
        "2",
        "3",
        "4",
    }
    assert all("duration" in track for item in first.values() for track in item["candidates"])

    invalid = _fma_frame().drop(columns=[("track", "duration")])
    with pytest.raises(ValueError, match="duration"):
        validate_fma_medium_metadata(invalid, audio_root)


def test_json_bytes_helper_matches_canonical_json_for_nfd_unicode():
    """Companion regression for the same real bug in the sibling writer that
    produces track_processing/selected_tracks.json, which is validated by
    the identical canonical-encoding check and draws from the same real FMA
    title/artist text.
    """
    from src.scripts.robust_track_processing import _json_bytes
    from src.utils.provenance import canonical_json_bytes

    payload = {"title": "5.La De\u0301cadence Orchestre\u0301e", "id": "141990"}
    assert _json_bytes(payload) == canonical_json_bytes(payload) + b"\n"


def test_export_population_file_is_byte_identical_to_canonical_json(tmp_path):
    """Regression for a real R1V release failure: a real FMA track title
    ("5.La De\u0301cadence Orchestre\u0301e", NFD-decomposed unicode) made
    export_population_file's independent json.dumps call diverge byte-for-byte
    from validate_release._read_canonical_json's canonical_json_bytes check,
    aborting the release at the final seal stage after the entire scientific
    pipeline had already completed. export_population_file must use
    canonical_json_bytes directly so the two can never drift apart again.
    """
    from src.scripts.generate_synthetic_users import (
        build_synthetic_population,
        export_population_file,
    )
    from src.utils.provenance import canonical_json_bytes

    tracks = _canonical_tracks(1000)
    tracks[0]["title"] = "5.La De\u0301cadence Orchestre\u0301e"
    population = build_synthetic_population(tracks)
    path = tmp_path / "population.json"

    export_population_file(population, path)

    raw = path.read_bytes()
    assert raw == canonical_json_bytes(population) + b"\n"


def test_validate_fma_medium_metadata_excludes_single_bad_row_rather_than_aborting(tmp_path):
    """A track with an unusable metadata value (for example the real FMA
    Medium metadata's one placeholder "unknown" title) must be excluded from
    the candidate pool as a structured admission failure, matching the
    documented genre-stratified quota rule's "failed track is replaced"
    behaviour, rather than aborting metadata loading for all 17,000 Medium
    tracks over a single bad row.
    """
    from src.scripts.robust_track_processing import validate_fma_medium_metadata

    audio_root = tmp_path / "audio"
    for track_id in range(1, 5):
        path = audio_root / "000" / f"{track_id:06d}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")

    frame = _fma_frame()
    frame.loc[2, ("track", "title")] = "Unknown"

    records = validate_fma_medium_metadata(frame, audio_root)

    assert {record["track_id"] for record in records} == {"1", "3", "4"}


def test_official_track_processing_rejects_noncanonical_target_before_io(tmp_path):
    from src.scripts.robust_track_processing import robust_track_processing

    with pytest.raises(ValueError, match="exactly 4000"):
        robust_track_processing(
            tmp_path / "missing.csv",
            3999,
            2025,
            tmp_path / "audio",
            tmp_path / "output",
            n_jobs=1,
        )


def _selected_payload() -> dict:
    return {
        "schema_version": 2,
        "record_type": "selected_fma_medium_tracks",
        "provenance": {
            "fma_metadata_sha256": "a" * 64,
            "fma_subset": "medium",
            "selection_seed": 2025,
            "selection_rule": "up-to-100-per-genre then Hamilton remaining capacity",
            "replacement_rule": "same genre before deterministic global capacity",
            "requested_tracks": 1,
            "genre_quotas": {"Rock": 1},
        },
        "tracks": [
            {
                "track_id": "1",
                "title": "Track one",
                "artist": "Artist one",
                "genre": "Rock",
                "duration": 120.0,
                "audio_relative_path": "000/000001.mp3",
            }
        ],
    }


def test_selected_track_loader_validates_exact_schema_and_provenance(tmp_path):
    from src.scripts.robust_track_processing import load_selected_tracks

    path = tmp_path / "selected_tracks.json"
    payload = _selected_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = load_selected_tracks(path, expected_count=1)
    assert records == payload["tracks"]

    payload["provenance"]["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_selected_tracks(path, expected_count=1)


def test_synthetic_generator_consumes_the_strict_selected_wrapper(tmp_path):
    from src.scripts.generate_synthetic_users import load_tracks_data

    path = tmp_path / "selected_tracks.json"
    path.write_text(json.dumps(_selected_payload()), encoding="utf-8")
    assert load_tracks_data(path, expected_count=1) == _selected_payload()["tracks"]


def test_synthetic_generator_cli_passes_the_official_selected_count(tmp_path, monkeypatch):
    import src.scripts.generate_synthetic_users as script

    tracks = _tracks(1)
    captured = {}

    def fake_load(path, *, expected_count):
        captured["expected_count"] = expected_count
        return tracks

    monkeypatch.setattr(script, "load_tracks_data", fake_load)
    monkeypatch.setattr(script, "validate_tracks_data", lambda value: None)
    monkeypatch.setattr(script, "build_synthetic_population", lambda *args, **kwargs: {})
    monkeypatch.setattr(script, "export_population_file", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(script, "print_summary", lambda *args, **kwargs: None)
    script.main(
        [
            "--tracks-json",
            str(tmp_path / "selected.json"),
            "--output-dir",
            str(tmp_path / "population"),
        ]
    )
    assert captured["expected_count"] == 4000


def test_all_signature_arm_admission_hook_checks_every_configuration(monkeypatch):
    import src.scripts.robust_track_processing as processing

    configurations = (
        SimpleNamespace(
            config_id="one", order=1, channels=("time", "pitch"), signature_dimension=3
        ),
        SimpleNamespace(
            config_id="two",
            order=2,
            channels=("time", "pitch", "loudness"),
            signature_dimension=13,
        ),
    )
    constructed = []
    admitted = []

    class FakeSignature:
        def __init__(self, order, expected_channels, **_kwargs):
            self.order = order
            self.expected_channels = expected_channels
            constructed.append((order, expected_channels))

        @staticmethod
        def get_signature_length_for_order(order, dimensions):
            return 1 + sum(dimensions**level for level in range(1, order + 1))

        def compute_signature(self, path, *, track_id):
            admitted.append((track_id, self.order, path.shape[1]))
            length = self.get_signature_length_for_order(
                self.order, self.expected_channels
            )
            return np.ones(length)

    monkeypatch.setattr(processing, "PathSignature", FakeSignature)
    hook = processing.make_all_signature_arms_admission_hook(configurations)
    # The order-2 three-channel signature contains the lower-order signature
    # of its ordered two-channel projection, so one numerical superset check
    # proves both declared arms while still validating both configurations.
    assert constructed == [(2, 3)]
    hook("7", _compact_feature(7, 4))
    assert admitted == [("7", 2, 3)]


def test_parallel_batch_runs_declared_signature_admission_inside_workers(monkeypatch):
    import src.scripts.robust_track_processing as processing

    worker_calls = []
    initialised_with = []

    class DeclaredHook:
        configuration_records = (
            {
                "config_id": "order_1__core",
                "order": 1,
                "channels": ["time", "pitch", "loudness"],
                "signature_dimension": 4,
            },
        )

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("declared admission must not be rerun in the parent")

    def fake_initialise(records):
        initialised_with.append(records)

        def worker_hook(track_id, _features):
            worker_calls.append(track_id)

        processing._WORKER_ADMISSION_HOOK = worker_hook

    class FakePool:
        def __init__(self, processes, *, initializer=None, initargs=()):
            assert processes == 2
            assert initializer is fake_initialise
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            processing._WORKER_ADMISSION_HOOK = None

        def map(self, function, tracks):
            outcomes = [function(track) for track in tracks]
            for outcome in outcomes:
                processing._WORKER_ADMISSION_HOOK(
                    outcome["track_id"], outcome["features"]
                )
            return outcomes

    monkeypatch.setattr(processing, "_initialise_worker_admission", fake_initialise)
    monkeypatch.setattr(processing, "Pool", FakePool)
    monkeypatch.setattr(
        processing,
        "process_audio_file",
        lambda track: {
            "status": "success",
            "track_id": track["track_id"],
            "features": _compact_feature(int(track["track_id"]), 3),
        },
    )
    tracks = [
        {"track_id": "1", "genre": "Rock"},
        {"track_id": "2", "genre": "Jazz"},
    ]
    outcomes = processing.process_tracks_batch(
        tracks, 2, admission_hook=DeclaredHook()
    )
    assert [row["status"] for row in outcomes] == ["success", "success"]
    assert worker_calls == ["1", "2"]
    assert initialised_with == [DeclaredHook.configuration_records]


def test_official_track_processing_cli_injects_all_signature_arms(tmp_path, monkeypatch):
    import src.scripts.robust_track_processing as processing

    configurations = tuple(SimpleNamespace(config_id=f"arm-{index}") for index in range(18))
    experiment_config = ModuleType("src.experiment_config")
    experiment_config.PATH_SELECTION_CONFIGS = configurations
    monkeypatch.setitem(sys.modules, "src.experiment_config", experiment_config)
    sentinel = object()
    monkeypatch.setattr(
        processing, "make_all_signature_arms_admission_hook", lambda value: sentinel
    )
    captured = {}

    def fake_process(*args, **kwargs):
        captured.update(kwargs)
        return [], {}, []

    monkeypatch.setattr(processing, "robust_track_processing", fake_process)

    class FakeTiming:
        def __init__(self, _name):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def save_report(self, _path):
            pass

        def print_summary(self):
            pass

    monkeypatch.setattr(processing, "TimingReport", FakeTiming)
    processing.main(
        [
            "--tracks-csv",
            str(tmp_path / "tracks.csv"),
            "--audio-root",
            str(tmp_path / "audio"),
            "--output-dir",
            str(tmp_path / "output"),
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )
    assert captured["admission_hook"] is sentinel


def test_strict_eda_uses_id_aligned_72_rows_and_true_per_genre_statistics(tmp_path):
    from src.analysis.strict_eda import run_strict_eda

    bundle_module = _feature_bundle_module()
    write_feature_bundle = bundle_module.write_feature_bundle

    tracks = _tracks(4)
    features = {}
    for index, track in enumerate(tracks):
        record = _compact_feature(index, 2 + index)
        record["traditional_feature_vector"] = np.full(72, float(index))
        features[str(track["track_id"])] = record
    bundle_path = tmp_path / "bundle"
    write_feature_bundle(features, bundle_path)
    output = tmp_path / "eda"

    results = run_strict_eda(tracks, bundle_path, output, expected_track_count=4)

    assert results["dataset_statistics"]["total_tracks"] == 4
    assert results["dataset_statistics"]["total_features"] == 72
    assert results["dataset_statistics"]["missing_values"] == 0
    from src.utils.provenance import canonical_json_bytes

    ordered_ids = sorted(str(track["track_id"]) for track in tracks)
    assert results["dataset_statistics"]["ordered_track_ids_sha256"] == (
        hashlib.sha256(canonical_json_bytes(ordered_ids)).hexdigest()
    )
    assert results["genre_statistics"]["Rock"]["feature_means"][0] == 0.5
    assert results["genre_statistics"]["Jazz"]["feature_means"][0] == 2.5
    correlation = np.genfromtxt(
        output / "correlation_matrix.csv", delimiter=",", skip_header=1
    )
    assert correlation.shape == (72, 73)
    missing = (output / "missing_value_outlier_summary.csv").read_text(
        encoding="utf-8"
    )
    assert "missing_count,missing_pct" in missing
    assert all(
        row.split(",")[1:3] == ["0", "0.0"]
        for row in missing.splitlines()[1:]
    )
    assert (output / "correlation_matrix.png").is_file()
    assert (output / "missing_value_outlier_summary.png").is_file()


def test_strict_eda_cli_passes_the_validated_track_list(tmp_path, monkeypatch):
    import src.scripts.generate_eda_analysis as script

    tracks = _tracks(1)
    captured = {}
    monkeypatch.setattr(script, "load_selected_tracks", lambda *args, **kwargs: tracks)

    def fake_eda(received, *args, **kwargs):
        captured["tracks"] = received

    monkeypatch.setattr(script, "run_strict_eda", fake_eda)
    result = script.main(
        [
            "--tracks-json",
            str(tmp_path / "selected.json"),
            "--feature-bundle",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "eda"),
        ]
    )
    assert result == 0
    assert captured["tracks"] is tracks


def test_interaction_heatmap_masks_empty_cells_and_labels_them_na(tmp_path):
    from src.scripts.plot_canonical_synthetic_users import (
        _build_interaction_matrix,
        _plot_interaction_heatmap,
    )

    rows = [
        ("casual_listener", "Rock", 1.0),
        ("explorer", "Jazz", 4.0),
    ]
    matrix, counts, archetypes, genres = _build_interaction_matrix(rows)
    casual = archetypes.index("casual_listener")
    explorer = archetypes.index("explorer")
    rock = genres.index("Rock")
    jazz = genres.index("Jazz")
    assert matrix[casual, rock] == 1.0
    assert not np.ma.getmaskarray(matrix)[casual, rock]
    assert counts[casual, jazz] == 0
    assert np.ma.getmaskarray(matrix)[casual, jazz]
    assert np.ma.getmaskarray(matrix)[explorer, rock]

    result = _plot_interaction_heatmap(rows, tmp_path / "heatmap.png")
    assert result["empty_cell_label"] == "N/A (no interactions)"
    assert result["empty_cell_count"] == 2
