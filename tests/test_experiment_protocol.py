"""Contract tests for canonical experiment task construction."""

import hashlib
import random

import numpy as np
import pytest
import src.evaluation.experiment_protocol as protocol

from src.evaluation.experiment_protocol import (
    COLD_START_SELECTION_DOMAIN,
    MINIMUM_CANDIDATES,
    ProtocolError,
    build_candidate_ids,
    build_index_map,
    derive_per_user_seed,
    normalise_and_sort_ids,
    normalise_id,
    select_cold_start_track_ids,
    split_user_interactions,
)


def test_id_normalisation_and_ascending_order_are_explicit():
    """Numeric IDs are canonical decimal strings; ordering is lexical."""

    assert normalise_id(42, kind="track") == "42"
    assert normalise_id(" 00042 ", kind="track") == "42"
    assert normalise_id(" user_02 ", kind="user") == "user_02"
    assert normalise_and_sort_ids(["10", "2", "001"], kind="track") == (
        "1",
        "10",
        "2",
    )
    assert build_index_map(["2", "001", "10"], kind="track") == {
        "1": 0,
        "10": 1,
        "2": 2,
    }


@pytest.mark.parametrize("value", [None, True, "", "   ", -1, "-1"])
def test_invalid_ids_fail_loudly(value):
    with pytest.raises(ProtocolError):
        normalise_id(value, kind="track")


def test_ids_that_collide_after_normalisation_are_rejected():
    with pytest.raises(ProtocolError, match="duplicate"):
        normalise_and_sort_ids([1, "001"], kind="track")


def test_per_user_split_matches_the_independent_sha256_oracle():
    interaction_ids = [str(value) for value in range(20, 0, -1)]

    split = split_user_interactions(
        " user_7 ", interaction_ids, master_seed=2025
    )

    sorted_ids = tuple(sorted(str(value) for value in range(1, 21)))
    seed = int.from_bytes(
        hashlib.sha256(b"2025:user_7").digest()[:8], "big", signed=False
    )
    permuted = tuple(np.random.default_rng(seed).permutation(sorted_ids).tolist())

    assert split.test == permuted[:3]
    assert split.validation == permuted[3:6]
    assert split.train == permuted[6:]
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert set(split.train) | set(split.validation) | set(split.test) == set(
        sorted_ids
    )


def test_per_user_split_is_invariant_to_input_order():
    ascending = [str(value) for value in range(1, 21)]
    descending = list(reversed(ascending))

    assert split_user_interactions("user_7", ascending) == split_user_interactions(
        "user_7", descending
    )


def test_per_user_split_does_not_mutate_process_global_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    split_user_interactions("user_7", [str(value) for value in range(1, 21)])

    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


def test_per_user_split_rejects_an_empty_partition():
    with pytest.raises(ProtocolError, match="empty"):
        split_user_interactions("user_7", [str(value) for value in range(1, 7)])


def test_cold_start_selection_matches_the_independent_sha256_oracle():
    catalogue = [str(value) for value in range(200, 0, -1)]

    cold = select_cold_start_track_ids(catalogue, master_seed=2025, cold_fraction=0.15)

    sorted_ids = tuple(sorted(str(value) for value in range(1, 201)))
    seed = int.from_bytes(
        hashlib.sha256(
            f"2025:{COLD_START_SELECTION_DOMAIN}".encode("utf-8")
        ).digest()[:8],
        "big",
        signed=False,
    )
    permuted = tuple(np.random.default_rng(seed).permutation(sorted_ids).tolist())
    expected = tuple(sorted(permuted[: int(0.15 * 200)]))

    assert cold == expected
    assert len(cold) == 30


def test_cold_start_selection_is_deterministic_and_order_invariant():
    ascending = [str(value) for value in range(1, 201)]
    descending = list(reversed(ascending))

    first = select_cold_start_track_ids(ascending, master_seed=2025, cold_fraction=0.15)
    second = select_cold_start_track_ids(ascending, master_seed=2025, cold_fraction=0.15)
    third = select_cold_start_track_ids(descending, master_seed=2025, cold_fraction=0.15)

    assert first == second == third
    assert list(first) == sorted(first)


def test_cold_start_selection_differs_from_per_user_split_seed():
    # The cold-start selection seed must come from the same derivation
    # function as per-user seeds, but keyed by its own fixed domain tag, not
    # collide with any real user ID's derived seed.
    cold_seed = derive_per_user_seed(2025, COLD_START_SELECTION_DOMAIN)
    user_seed = derive_per_user_seed(2025, "user_7")
    assert cold_seed != user_seed


def test_cold_start_selection_uses_a_reasonable_default_fraction():
    catalogue = [str(value) for value in range(1, 101)]
    cold = select_cold_start_track_ids(catalogue, master_seed=2025)
    assert len(cold) == 15  # default cold_fraction=0.15, matches 70/15/15 convention
    assert len(catalogue) - len(cold) >= MINIMUM_CANDIDATES


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_cold_start_selection_rejects_out_of_range_fractions(fraction):
    catalogue = [str(value) for value in range(1, 101)]
    with pytest.raises(ProtocolError, match="cold_fraction"):
        select_cold_start_track_ids(catalogue, cold_fraction=fraction)


def test_cold_start_selection_rejects_a_fraction_too_small_to_select_anything():
    catalogue = [str(value) for value in range(1, 10)]
    with pytest.raises(ProtocolError, match="too small"):
        select_cold_start_track_ids(catalogue, cold_fraction=0.05)


def test_cold_start_selection_rejects_leaving_too_few_warm_tracks():
    catalogue = [str(value) for value in range(1, 12)]
    with pytest.raises(ProtocolError, match="warm tracks"):
        select_cold_start_track_ids(catalogue, cold_fraction=0.5)


def test_candidate_ids_are_sorted_unobserved_catalogue_ids():
    expected = ("10", "11", "12", "3", "4", "5", "6", "7", "8", "9")

    assert build_candidate_ids(
        catalogue_ids=[
            "12",
            "11",
            "10",
            "9",
            "8",
            "7",
            "6",
            "5",
            "4",
            "03",
            "2",
            "1",
        ],
        observed_ids=["2", "1"],
    ) == expected
    assert build_candidate_ids(
        catalogue_ids=[
            "1",
            "2",
            "03",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
        ],
        observed_ids=["1", "2"],
    ) == expected


def test_candidate_construction_requires_at_least_ten_candidates():
    with pytest.raises(ProtocolError, match="at least 10"):
        build_candidate_ids(
            catalogue_ids=[str(value) for value in range(1, 12)],
            observed_ids=["1", "2"],
        )


def test_candidate_construction_rejects_unknown_or_duplicate_ids():
    with pytest.raises(ProtocolError, match="unknown"):
        build_candidate_ids(catalogue_ids=["1", "2"], observed_ids=["3"])

    with pytest.raises(ProtocolError, match="duplicate"):
        build_candidate_ids(catalogue_ids=[1, "001", "2"], observed_ids=[])


def test_evaluation_unit_records_only_in_catalogue_test_relevance():
    unit = protocol.build_evaluation_unit(
        catalogue_ids=[str(value) for value in range(1, 13)],
        observed_ids=["1", "2"],
        test_ids=["3", "99"],
    )

    assert unit.candidate_ids == (
        "10",
        "11",
        "12",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    )
    assert unit.relevance_ids == ("3",)
    assert unit.out_of_catalogue_test_ids == ("99",)


def test_evaluation_unit_requires_an_in_catalogue_test_interaction():
    with pytest.raises(ProtocolError, match="in-catalogue test interaction"):
        protocol.build_evaluation_unit(
            catalogue_ids=[str(value) for value in range(1, 13)],
            observed_ids=["1", "2"],
            test_ids=["99"],
        )


def test_evaluation_unit_rejects_observed_test_overlap():
    with pytest.raises(ProtocolError, match="must be disjoint"):
        protocol.build_evaluation_unit(
            catalogue_ids=[str(value) for value in range(1, 13)],
            observed_ids=["1", "2"],
            test_ids=["2", "3"],
        )


def test_index_map_manifest_is_lexical_checksummed_and_order_invariant():
    first = protocol.build_index_maps(
        user_ids=["user_2", "user_1"], item_ids=["2", "10", "001"]
    )
    second = protocol.build_index_maps(
        user_ids=["user_1", "user_2"], item_ids=["001", "10", "2"]
    )

    assert first == second
    assert first.users == {"user_1": 0, "user_2": 1}
    assert first.items == {"1": 0, "10": 1, "2": 2}
    assert len(first.users_checksum) == 64
    assert len(first.items_checksum) == 64

    with pytest.raises(ProtocolError, match="duplicate"):
        protocol.build_index_maps(user_ids=["user_1", " user_1 "], item_ids=["1"])


def test_index_map_properties_return_fresh_copies():
    maps = protocol.build_index_maps(user_ids=["user_1"], item_ids=["track_1"])
    users = maps.users
    items = maps.items
    users["user_2"] = 1
    items["track_2"] = 1

    assert maps.users == {"user_1": 0}
    assert maps.items == {"track_1": 0}
