# pylint: disable=broad-except
"""
Robust track processing script that handles track selection and audio processing with retry mechanism.

This script ensures we get the requested number of valid tracks by:
1. Selecting tracks from FMA dataset
2. Processing audio features
3. If some tracks fail, selecting additional tracks
4. Repeating until we have enough valid tracks
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import TimingReport
from src.audio.feature_extraction import AudioFeatureExtractor


def get_audio_path(track_id, audio_root):
    """Get the audio path for a given track ID."""
    tid = int(track_id)
    return os.path.join(audio_root, f"{tid:06d}"[:3], f"{tid:06d}.mp3")


def select_initial_tracks(tracks_csv, n_tracks, seed, audio_root, logger):
    """Select initial tracks from the FMA dataset with balanced genre sampling.

    Implements stratified sampling with minimum threshold:
    - Minimum 100 tracks per genre (where available)
    - Genres with <100 tracks available use all available tracks
    - Remaining slots distributed proportionally among major genres
    """
    # Load tracks metadata
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    available = tracks[("set", "subset")] == "medium"
    tracks = tracks[available]

    # Check available tracks
    available_count = len(tracks)
    logger.info("Number of tracks in medium subset: %d", available_count)

    # Use all available tracks if requesting more than available
    if n_tracks > available_count:
        logger.warning(
            "Requested %d tracks but only %d available. Using all available tracks.",
            n_tracks,
            available_count,
        )
        n_tracks = available_count

    # For memory efficiency, limit to a reasonable number
    max_tracks = 5000  # Adjust based on your memory
    if n_tracks > max_tracks:
        logger.warning("Limiting to %d tracks for memory efficiency.", max_tracks)
        n_tracks = max_tracks

    # Ensure n_tracks is positive
    if n_tracks <= 0:
        logger.error("Invalid n_tracks: %d. Must be positive.", n_tracks)
        return []

    # Initialize selected_tracks
    selected_tracks = None

    # Group tracks by genre
    genre_column = ("track", "genre_top")
    if genre_column not in tracks.columns:
        logger.warning("Genre column not found, falling back to random selection")
        rng = np.random.default_rng(seed)
        selected = rng.choice(tracks.index, size=n_tracks, replace=False)
        selected_tracks = tracks.loc[selected]
    else:
        # Group by genre
        genre_groups = tracks.groupby(genre_column)
        genre_counts = genre_groups.size()

        # Filter out genres with no tracks (safety check)
        genre_counts = genre_counts[genre_counts > 0]

        if len(genre_counts) == 0:
            logger.warning(
                "No genres found with tracks, falling back to random selection"
            )
            rng = np.random.default_rng(seed)
            selected = rng.choice(tracks.index, size=n_tracks, replace=False)
            selected_tracks = tracks.loc[selected]
        else:
            logger.info("Genre distribution in medium subset:")
            for genre, count in genre_counts.items():
                logger.info("  %s: %d tracks", genre, count)

            # Minimum threshold per genre
            min_per_genre = 100

            # Identify minor genres (< min_per_genre available)
            minor_genres = genre_counts[genre_counts < min_per_genre].index.tolist()
            major_genres = genre_counts[genre_counts >= min_per_genre].index.tolist()

            # Initialize allocations
            minor_allocations = {}
            major_allocations = {}
            minor_total = 0

            # Defensive check: ensure we have genres to work with
            if len(minor_genres) == 0 and len(major_genres) == 0:
                logger.warning(
                    "No genres found after filtering, falling back to random selection"
                )
                rng = np.random.default_rng(seed)
                selected = rng.choice(tracks.index, size=n_tracks, replace=False)
                selected_tracks = tracks.loc[selected]
            else:
                # Allocate tracks for minor genres (use all available)
                for genre in minor_genres:
                    count = int(genre_counts[genre])
                    if count > 0:  # Safety check
                        minor_allocations[genre] = count
                        minor_total += count
                        logger.info(
                            "Minor genre '%s': allocating all %d available tracks",
                            genre,
                            count,
                        )

                # Calculate remaining slots for major genres
                remaining_slots = n_tracks - minor_total

                if remaining_slots < 0:
                    logger.warning(
                        "Minor genres total (%d) exceeds requested tracks (%d). "
                        "Using all minor genre tracks and reducing total.",
                        minor_total,
                        n_tracks,
                    )
                    remaining_slots = 0

                # Allocate minimum to each major genre
                major_min_total = (
                    len(major_genres) * min_per_genre if major_genres else 0
                )

                if remaining_slots < major_min_total:
                    logger.warning(
                        "Remaining slots (%d) insufficient for minimum allocation (%d). "
                        "Distributing proportionally without minimum guarantee.",
                        remaining_slots,
                        major_min_total,
                    )
                    # Proportional distribution without minimum
                    major_total_available = (
                        genre_counts[major_genres].sum() if major_genres else 0
                    )
                    major_allocations = {}
                    if (
                        major_total_available > 0 and len(major_genres) > 0
                    ):  # Safety check for division by zero
                        for genre in major_genres:
                            proportion = genre_counts[genre] / major_total_available
                            allocation = max(1, int(remaining_slots * proportion))
                            major_allocations[genre] = allocation
                    elif len(major_genres) > 0:
                        # Fallback: equal distribution
                        tracks_per_genre = max(1, remaining_slots // len(major_genres))
                        major_allocations = {
                            genre: tracks_per_genre for genre in major_genres
                        }
                    else:
                        # No major genres, empty allocation
                        major_allocations = {}
                else:
                    # Allocate minimum to each major genre
                    major_allocations = {genre: min_per_genre for genre in major_genres}
                    remaining_after_min = remaining_slots - major_min_total

                    # Distribute remaining slots proportionally
                    if remaining_after_min > 0 and len(major_genres) > 0:
                        major_total_available = genre_counts[major_genres].sum()
                        if major_total_available > 0:  # Safety check
                            for genre in major_genres:
                                proportion = genre_counts[genre] / major_total_available
                                additional = int(remaining_after_min * proportion)
                                major_allocations[genre] += additional
                        else:
                            # Fallback: equal distribution
                            tracks_per_genre = (
                                remaining_after_min // len(major_genres)
                                if major_genres
                                else 0
                            )
                            for genre in major_genres:
                                major_allocations[genre] += tracks_per_genre

                        # Adjust for rounding errors (with safety limit)
                        allocated_total = sum(major_allocations.values()) + minor_total
                        if allocated_total < n_tracks:
                            # Distribute remaining to largest genres (with max iterations to prevent infinite loop)
                            remaining = n_tracks - allocated_total
                            sorted_genres = sorted(
                                major_genres,
                                key=lambda g: genre_counts[g],
                                reverse=True,
                            )
                            # Limit distribution to prevent infinite loops
                            max_distributions = min(remaining, len(sorted_genres))
                            for genre in sorted_genres[:max_distributions]:
                                major_allocations[genre] += 1

                # Combine allocations
                all_allocations = {**minor_allocations, **major_allocations}

                # Defensive check: ensure we have allocations
                if not all_allocations:
                    logger.warning(
                        "No genre allocations made, falling back to random selection"
                    )
                    rng = np.random.default_rng(seed)
                    selected = rng.choice(tracks.index, size=n_tracks, replace=False)
                    selected_tracks = tracks.loc[selected]
                else:
                    logger.info("Final genre allocations:")
                    for genre, allocation in sorted(
                        all_allocations.items(), key=lambda x: x[1], reverse=True
                    ):
                        available = int(genre_counts[genre])
                        logger.info(
                            "  %s: %d tracks (out of %d available)",
                            genre,
                            allocation,
                            available,
                        )

                    # Randomly sample within each genre quota
                    rng = np.random.default_rng(seed)
                    selected_indices = []

                    for genre, quota in all_allocations.items():
                        try:
                            genre_tracks = genre_groups.get_group(genre)
                            available_in_genre = len(genre_tracks)

                            if available_in_genre == 0:
                                logger.warning(
                                    "Genre '%s' has no tracks available, skipping",
                                    genre,
                                )
                                continue

                            if quota > available_in_genre:
                                logger.warning(
                                    "Requested %d tracks for genre '%s' but only %d available. Using all available.",
                                    quota,
                                    genre,
                                    available_in_genre,
                                )
                                quota = available_in_genre

                            if quota > 0:
                                genre_indices = genre_tracks.index.tolist()
                                if len(genre_indices) < quota:
                                    logger.warning(
                                        "Genre '%s' has fewer indices (%d) than quota (%d), using all available",
                                        genre,
                                        len(genre_indices),
                                        quota,
                                    )
                                    quota = len(genre_indices)

                                if quota > 0:
                                    # Defensive check: ensure we have indices to choose from
                                    if len(genre_indices) == 0:
                                        logger.warning(
                                            "Genre '%s' has no indices available", genre
                                        )
                                        continue

                                    selected_genre_indices = rng.choice(
                                        genre_indices, size=quota, replace=False
                                    )
                                    # Handle case where rng.choice returns scalar
                                    if isinstance(
                                        selected_genre_indices, (int, np.integer)
                                    ):
                                        selected_indices.append(selected_genre_indices)
                                    else:
                                        selected_indices.extend(
                                            selected_genre_indices.tolist()
                                        )
                        except KeyError:
                            logger.warning(
                                "Genre '%s' not found in genre groups, skipping", genre
                            )
                            continue
                        except Exception as e:
                            logger.error("Error sampling genre '%s': %s", genre, e)
                            continue

                    # Ensure we have selected tracks
                    if not selected_indices:
                        logger.error(
                            "No tracks selected! Falling back to random selection"
                        )
                        rng = np.random.default_rng(seed)
                        selected = rng.choice(
                            tracks.index, size=min(n_tracks, len(tracks)), replace=False
                        )
                        selected_tracks = tracks.loc[selected]
                    else:
                        # Remove duplicates (defensive check)
                        selected_indices = list(set(selected_indices))
                        selected_tracks = tracks.loc[selected_indices]

                        # Verify we got the right number
                        actual_count = len(selected_tracks)
                        if actual_count != n_tracks:
                            logger.info(
                                "Selected %d tracks (requested %d) due to genre balancing constraints",
                                actual_count,
                                n_tracks,
                            )

    # Defensive check: ensure selected_tracks is initialized
    if selected_tracks is None:
        logger.error(
            "selected_tracks was never initialized! Falling back to random selection"
        )
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            tracks.index, size=min(n_tracks, len(tracks)), replace=False
        )
        selected_tracks = tracks.loc[selected]

    # Build output list
    output = []
    for idx, row in selected_tracks.iterrows():
        # Handle case where genre column might not exist
        genre_value = None
        if genre_column in tracks.columns:
            try:
                genre_value = row[genre_column]
            except (KeyError, IndexError):
                genre_value = None

        info = {
            "track_id": int(idx),
            "title": row[("track", "title")],
            "artist": row[("artist", "name")],
            "genre": genre_value,
            "file_path": get_audio_path(idx, audio_root),
        }
        output.append(info)

    logger.info("Seed: %d", seed)
    logger.info("Total selected tracks: %d", len(output))
    return output


def select_additional_tracks(
    tracks_csv, n_additional, seed, audio_root, existing_track_ids, logger
):
    """Select additional tracks that haven't been processed yet."""
    # Load tracks metadata
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    available = tracks[("set", "subset")] == "medium"
    tracks = tracks[available]

    # Filter out already processed tracks
    unprocessed_tracks = tracks[~tracks.index.isin(existing_track_ids)]

    if len(unprocessed_tracks) < n_additional:
        logger.warning(
            f"Only {len(unprocessed_tracks)} unprocessed tracks available, but {n_additional} requested"
        )
        n_additional = len(unprocessed_tracks)

    if n_additional == 0:
        return []

    # Random selection
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        unprocessed_tracks.index, size=n_additional, replace=False
    )
    selected_tracks = unprocessed_tracks.loc[selected_indices]

    # Build output list
    additional_tracks = []
    for idx, row in selected_tracks.iterrows():
        info = {
            "track_id": int(idx),
            "title": row[("track", "title")],
            "artist": row[("artist", "name")],
            "genre": row[("track", "genre_top")],
            "file_path": get_audio_path(idx, audio_root),
        }
        additional_tracks.append(info)

    logger.info(f"Selected {len(additional_tracks)} additional tracks")
    return additional_tracks


def process_audio_file(track):
    """Process a single audio file and return its features."""
    logger = setup_logger("worker")
    extractor = AudioFeatureExtractor()
    audio_file = track["file_path"]
    label = str(track["track_id"])

    try:
        logger.info("Processing %s", audio_file)
        features = extractor.extract_features(audio_file)

        # Validate features
        if not features:
            logger.error("No features extracted from %s", audio_file)
            return None, None

        # Check for empty or invalid features
        for key, value in features.items():
            if isinstance(value, np.ndarray) and (
                value.size == 0 or np.isnan(value).any()
            ):
                logger.error("Invalid features found in %s for %s", audio_file, key)
                return None, None

        return features, label

    except Exception as e:
        logger.error("Error processing %s: %s", audio_file, str(e))
        return None, None


def process_tracks_batch(tracks, n_jobs, logger):
    """Process a batch of tracks and return results."""
    if not tracks:
        return [], []

    logger.info(f"Processing batch of {len(tracks)} tracks...")

    # Process tracks in parallel
    if n_jobs == 1:
        results = [process_audio_file(track) for track in tracks]
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(process_audio_file, tracks)

    # Process results
    valid_features = {}
    failed_tracks = []

    for (features, label), track in zip(results, tracks):
        if features is not None and label is not None:
            valid_features[label] = features
        else:
            failed_tracks.append(track)

    logger.info(
        f"Batch complete: {len(valid_features)} valid, {len(failed_tracks)} failed"
    )
    return valid_features, failed_tracks


def robust_track_processing(
    tracks_csv, n_tracks, seed, audio_root, output_dir, n_jobs=None, timing=None
):
    """Robust track processing with retry mechanism."""
    logger = setup_logger("robust_track_processing")

    # Set up parallel processing
    if n_jobs is None:
        n_jobs = min(cpu_count() - 1, 4)  # Limit to 4 to avoid memory issues

    logger.info("Using %d parallel processes", n_jobs)
    logger.info("Target: %d valid tracks", n_tracks)

    # Step 1: Select initial tracks
    section_name = "Initial Track Selection"
    if timing:
        with timing.section(section_name):
            logger.info("Step 1: Selecting initial tracks...")
            all_tracks = select_initial_tracks(
                tracks_csv, n_tracks, seed, audio_root, logger
            )
    else:
        logger.info("Step 1: Selecting initial tracks...")
        all_tracks = select_initial_tracks(
            tracks_csv, n_tracks, seed, audio_root, logger
        )

    if not all_tracks:
        logger.error("Failed to select initial tracks")
        return

    logger.info("Selected %d initial tracks", len(all_tracks))

    # Track processing state
    valid_features = {}
    failed_tracks = []
    processed_track_ids = set()
    all_processed_tracks = []  # Keep track of all processed tracks for final output
    batch_size = min(50, n_tracks)  # Process in smaller batches

    # Step 2: Process tracks in batches with retry mechanism
    section_name = "Track Processing (Audio Feature Extraction)"
    if timing:
        with timing.section(section_name):
            logger.info("Step 2: Processing tracks with retry mechanism...")
            _process_tracks_loop(
                all_tracks,
                n_tracks,
                n_jobs,
                batch_size,
                tracks_csv,
                seed,
                audio_root,
                valid_features,
                failed_tracks,
                processed_track_ids,
                all_processed_tracks,
                logger,
            )
    else:
        logger.info("Step 2: Processing tracks with retry mechanism...")
        _process_tracks_loop(
            all_tracks,
            n_tracks,
            n_jobs,
            batch_size,
            tracks_csv,
            seed,
            audio_root,
            valid_features,
            failed_tracks,
            processed_track_ids,
            all_processed_tracks,
            logger,
        )

    # Step 3: Save results
    section_name = "Results Saving"
    if timing:
        with timing.section(section_name):
            _save_results(
                output_dir,
                n_tracks,
                valid_features,
                all_processed_tracks,
                failed_tracks,
                processed_track_ids,
                seed,
                logger,
            )
    else:
        _save_results(
            output_dir,
            n_tracks,
            valid_features,
            all_processed_tracks,
            failed_tracks,
            processed_track_ids,
            seed,
            logger,
        )

    # Extract return values
    valid_track_ids = list(valid_features.keys())[:n_tracks]
    valid_tracks = []
    for track in all_processed_tracks:
        if str(track["track_id"]) in valid_track_ids:
            valid_tracks.append(track)
    valid_tracks = valid_tracks[:n_tracks]
    valid_features = {k: v for k, v in list(valid_features.items())[:n_tracks]}

    return valid_tracks, valid_features, failed_tracks


def _process_tracks_loop(
    all_tracks,
    n_tracks,
    n_jobs,
    batch_size,
    tracks_csv,
    seed,
    audio_root,
    valid_features,
    failed_tracks,
    processed_track_ids,
    all_processed_tracks,
    logger,
):
    """Process tracks in batches with retry mechanism."""
    while len(valid_features) < n_tracks and len(all_tracks) > 0:
        # Get next batch of tracks to process
        batch_tracks = all_tracks[:batch_size]
        all_tracks = all_tracks[batch_size:]

        # Process batch
        batch_features, batch_failed = process_tracks_batch(
            batch_tracks, n_jobs, logger
        )

        # Update state
        valid_features.update(batch_features)
        failed_tracks.extend(batch_failed)

        # Track processed IDs and add to all processed tracks
        for track in batch_tracks:
            processed_track_ids.add(track["track_id"])
            all_processed_tracks.append(track)

        logger.info("Progress: %d/%d valid tracks", len(valid_features), n_tracks)

        # If we need more tracks and have failed tracks, select additional ones
        if len(valid_features) < n_tracks and len(all_tracks) == 0:
            needed = n_tracks - len(valid_features)
            # Select extra tracks to account for potential failures
            extra_tracks = max(
                needed * 2, 50
            )  # Select 2x needed or 50, whichever is larger

            logger.info(
                "Need %d more tracks, selecting %d additional tracks...",
                needed,
                extra_tracks,
            )

            additional_tracks = select_additional_tracks(
                tracks_csv,
                extra_tracks,
                seed + len(processed_track_ids),
                audio_root,
                processed_track_ids,
                logger,
            )

            if not additional_tracks:
                logger.warning("No more tracks available for selection")
                break

            all_tracks.extend(additional_tracks)


def _save_results(
    output_dir,
    n_tracks,
    valid_features,
    all_processed_tracks,
    failed_tracks,
    processed_track_ids,
    seed,
    logger,
):
    """Save processing results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save valid tracks (first n_tracks)
    valid_track_ids = list(valid_features.keys())[:n_tracks]
    valid_tracks = []

    # Find valid tracks by ID from all processed tracks
    for track in all_processed_tracks:
        if str(track["track_id"]) in valid_track_ids:
            valid_tracks.append(track)

    # Trim to requested number
    valid_tracks = valid_tracks[:n_tracks]
    valid_features_trimmed = {k: v for k, v in list(valid_features.items())[:n_tracks]}

    # Save results
    tracks_output = os.path.join(output_dir, "selected_tracks.json")
    with open(tracks_output, "w", encoding="utf-8") as f:
        json.dump(valid_tracks, f, indent=2)

    features_output = os.path.join(output_dir, "features.json")
    extractor = AudioFeatureExtractor()
    extractor.save_features(valid_features_trimmed, features_output)

    # Save processing report
    report = {
        "requested_tracks": n_tracks,
        "valid_tracks": len(valid_tracks),
        "failed_tracks": len(failed_tracks),
        "total_processed": len(processed_track_ids),
        "success_rate": (
            len(valid_tracks) / len(processed_track_ids) if processed_track_ids else 0
        ),
        "failed_track_details": failed_tracks[:20],  # First 20 failures
        "seed": seed,
    }

    report_output = os.path.join(output_dir, "processing_report.json")
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("ROBUST TRACK PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Requested tracks: {n_tracks}")
    print(f"Valid tracks obtained: {len(valid_tracks)}")
    print(f"Failed tracks: {len(failed_tracks)}")
    print(f"Total processed: {len(processed_track_ids)}")
    print(f"Success rate: {report['success_rate']:.2%}")
    print(f"Tracks saved to: {tracks_output}")
    print(f"Features saved to: {features_output}")
    print(f"Report saved to: {report_output}")
    print(f"{'='*60}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Robust track processing: select tracks and process audio with retry mechanism."
    )
    parser.add_argument(
        "--tracks-csv",
        default="./data/fma_metadata/tracks.csv",
        help="Path to tracks.csv",
    )
    parser.add_argument(
        "--n-tracks",
        type=int,
        default=4000,
        help="Number of tracks to select (default: 4000)",
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed (default: 2025)"
    )
    parser.add_argument(
        "--audio-root",
        default="./data/fma_medium",
        help="Root directory for audio files",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/processed_tracks",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: auto)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--results-dir",
        default="./results/track_processing",
        help="Directory for logs and timing reports (default: ./results/track_processing)",
    )

    args = parser.parse_args()

    # Create results directory for logs and timing
    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Create output directory for data files
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up logging with file handler
    log_file = results_path / "robust_track_processing.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("robust_track_processing")
    logger.info("Logging to file: %s", log_file)

    # Generate seed if not provided
    seed = args.seed if args.seed is not None else int(datetime.now().timestamp())

    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize timing
    timing = TimingReport("robust_track_processing")
    timing.start()

    try:
        # Run robust track processing
        robust_track_processing(
            args.tracks_csv,
            args.n_tracks,
            seed,
            args.audio_root,
            args.output_dir,
            args.n_jobs,
            timing=timing,
        )

        # Stop timing and save report
        timing.stop()
        timing.save_report(results_path)
        timing.print_summary()
    except Exception as e:
        logger = setup_logger("robust_track_processing")
        logger.error("Robust track processing failed: %s", str(e))
        timing.stop()
        timing.save_report(results_path)
        raise


if __name__ == "__main__":
    main()
