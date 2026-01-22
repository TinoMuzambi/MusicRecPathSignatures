# pylint: disable=broad-except
"""
Test script for ablation studies to verify functionality with a small dataset.
"""

import numpy as np
from src.signatures.path_signatures import PathSignature
from src.analysis.softmax_regression import SoftmaxRegression
from src.evaluation import RecommendationMetrics


def create_test_features():
    """Create synthetic test features for ablation studies."""
    test_features = {}

    # Create 5 test songs with synthetic features
    for i in range(5):
        song_name = f"test_song_{i + 1}"

        # Create synthetic multi-dimensional time series
        # Simulate 10 time steps with 3 dimensions (MFCC-like features)
        time_series = np.random.randn(10, 3)

        # Add some structure to make songs similar within groups
        if i < 2:  # First group
            time_series += np.array([1.0, 0.5, -0.5])
        else:  # Second group
            time_series += np.array([-0.5, 1.0, 0.5])

        test_features[song_name] = {
            "multi_dimensional_series": time_series.tolist(),
            "mfccs": np.random.randn(13).tolist(),
            "chroma": np.random.randn(12).tolist(),
            "spectral_centroid": np.random.randn(1).tolist(),
            "spectral_bandwidth": np.random.randn(1).tolist(),
            "zero_crossing_rate": np.random.randn(1).tolist(),
        }

    return test_features


def test_path_signature_orders():
    """Test path signature order analysis."""
    print("Testing path signature order analysis...")

    features = create_test_features()

    # Test different orders
    for order in [1, 2, 3]:
        try:
            path_sig = PathSignature(order=order)
            signatures = path_sig.compute_signatures_dict(features)

            if signatures:
                print(f"✓ Order {order}: {len(signatures)} signatures computed")
                first_signature = list(signatures.values())[0]
                print(f"  Signature length: {len(first_signature)}")
            else:
                print(f"✗ Order {order}: No signatures computed")

        except Exception as e:
            print(f"✗ Order {order}: Error - {str(e)}")

    print()


def test_temperature_scaling():
    """Test temperature scaling analysis."""
    print("Testing temperature scaling analysis...")

    features = create_test_features()

    # Compute signatures once
    path_sig = PathSignature(order=2)
    signatures = path_sig.compute_signatures_dict(features)

    if not signatures:
        print("✗ No signatures computed for temperature test")
        return

    # Test different temperatures
    for temp in [0.5, 1.0, 2.0]:
        try:
            softmax_model = SoftmaxRegression()
            similarity_matrix, _ = softmax_model.compute_similarity_matrix(
                signatures, temperature=temp
            )

            print(
                f"✓ Temperature {temp:.1f}: Similarity matrix shape {similarity_matrix.shape}"
            )
            print(f"  Mean similarity: {np.mean(similarity_matrix):.4f}")
            print(f"  Std similarity: {np.std(similarity_matrix):.4f}")

        except Exception as e:
            print(f"✗ Temperature {temp:.1f}: Error - {str(e)}")

    print()


def test_feature_combinations():
    """Test feature combination analysis."""
    print("Testing feature combination analysis...")

    features = create_test_features()

    # Test different feature combinations
    combinations = [["mfccs"], ["chroma"], ["mfccs", "chroma"]]

    for combo in combinations:
        combo_name = "+".join(combo)
        try:
            # Filter features
            filtered_features = {}
            for song_name, song_features in features.items():
                filtered_features[song_name] = {}
                for feature_name in combo:
                    if feature_name in song_features:
                        filtered_features[song_name][feature_name] = song_features[
                            feature_name
                        ]

                # Keep multi_dimensional_series
                if "multi_dimensional_series" in song_features:
                    filtered_features[song_name]["multi_dimensional_series"] = (
                        song_features["multi_dimensional_series"]
                    )

            # Compute signatures
            path_sig = PathSignature(order=2)
            signatures = path_sig.compute_signatures_dict(filtered_features)

            if signatures:
                print(f"✓ {combo_name}: {len(signatures)} signatures computed")
            else:
                print(f"✗ {combo_name}: No signatures computed")

        except Exception as e:
            print(f"✗ {combo_name}: Error - {str(e)}")

    print()


def test_similarity_metrics():
    """Test similarity metric analysis."""
    print("Testing similarity metric analysis...")

    features = create_test_features()

    # Compute signatures
    path_sig = PathSignature(order=2)
    signatures = path_sig.compute_signatures_dict(features)

    if not signatures:
        print("✗ No signatures computed for similarity test")
        return

    # Test different similarity metrics
    metrics = ["cosine", "euclidean", "manhattan"]

    for metric in metrics:
        try:
            # Compute similarity matrix
            song_names_list = list(signatures.keys())
            n_songs = len(song_names_list)
            similarity_matrix = np.zeros((n_songs, n_songs))

            for i, song1 in enumerate(song_names_list):
                for j, song2 in enumerate(song_names_list):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        sig1 = signatures[song1]
                        sig2 = signatures[song2]

                        if metric == "cosine":
                            dot_product = np.dot(sig1, sig2)
                            norm1 = np.linalg.norm(sig1)
                            norm2 = np.linalg.norm(sig2)
                            similarity_matrix[i, j] = dot_product / (
                                norm1 * norm2 + 1e-8
                            )
                        elif metric == "euclidean":
                            distance = np.linalg.norm(sig1 - sig2)
                            similarity_matrix[i, j] = 1.0 / (1.0 + distance)
                        elif metric == "manhattan":
                            distance = np.sum(np.abs(sig1 - sig2))
                            similarity_matrix[i, j] = 1.0 / (1.0 + distance)

            print(f"✓ {metric}: Similarity matrix computed")
            print(f"  Mean similarity: {np.mean(similarity_matrix):.4f}")
            print(
                f"  Range: [{np.min(similarity_matrix):.4f}, {np.max(similarity_matrix):.4f}]"
            )

        except Exception as e:
            print(f"✗ {metric}: Error - {str(e)}")

    print()


def test_component_contributions():
    """Test component contribution analysis."""
    print("Testing component contribution analysis...")

    features = create_test_features()

    # Test different components
    components = {
        "path_signatures_only": "Path signatures without softmax",
        "softmax_only": "Softmax regression without temperature scaling",
        "temperature_scaling_only": "Temperature scaling without softmax",
        "hybrid_full": "Full hybrid approach",
    }

    for component_name, description in components.items():
        try:
            # Compute signatures
            path_sig = PathSignature(order=2)
            signatures = path_sig.compute_signatures_dict(features)

            if not signatures:
                print(f"✗ {component_name}: No signatures computed")
                continue

            if component_name == "path_signatures_only":
                # Simple cosine similarity
                song_names_list = list(signatures.keys())
                n_songs = len(song_names_list)
                similarity_matrix = np.zeros((n_songs, n_songs))

                for i, song1 in enumerate(song_names_list):
                    for j, song2 in enumerate(song_names_list):
                        if i == j:
                            similarity_matrix[i, j] = 1.0
                        else:
                            sig1 = signatures[song1]
                            sig2 = signatures[song2]
                            dot_product = np.dot(sig1, sig2)
                            norm1 = np.linalg.norm(sig1)
                            norm2 = np.linalg.norm(sig2)
                            similarity_matrix[i, j] = dot_product / (
                                norm1 * norm2 + 1e-8
                            )

            elif component_name == "softmax_only":
                # Softmax with temperature=1.0
                softmax_model = SoftmaxRegression()
                similarity_matrix, _ = softmax_model.compute_similarity_matrix(
                    signatures, temperature=1.0
                )

            elif component_name == "temperature_scaling_only":
                # Cosine similarity with temperature scaling
                song_names_list = list(signatures.keys())
                n_songs = len(song_names_list)
                similarity_matrix = np.zeros((n_songs, n_songs))

                for i, song1 in enumerate(song_names_list):
                    for j, song2 in enumerate(song_names_list):
                        if i == j:
                            similarity_matrix[i, j] = 1.0
                        else:
                            sig1 = signatures[song1]
                            sig2 = signatures[song2]
                            dot_product = np.dot(sig1, sig2)
                            norm1 = np.linalg.norm(sig1)
                            norm2 = np.linalg.norm(sig2)
                            similarity_matrix[i, j] = dot_product / (
                                norm1 * norm2 + 1e-8
                            )

                # Apply temperature scaling
                temperature = 2.0
                similarity_matrix = np.exp(similarity_matrix / temperature)
                similarity_matrix = similarity_matrix / np.sum(
                    similarity_matrix, axis=1, keepdims=True
                )

            elif component_name == "hybrid_full":
                # Full hybrid approach
                softmax_model = SoftmaxRegression()
                similarity_matrix, _ = softmax_model.compute_similarity_matrix(
                    signatures, temperature=2.0
                )

            print(f"✓ {component_name}: {description}")
            print(f"  Similarity matrix shape: {similarity_matrix.shape}")
            print(f"  Mean similarity: {np.mean(similarity_matrix):.4f}")

        except Exception as e:
            print(f"✗ {component_name}: Error - {str(e)}")

    print()


def test_evaluation_metrics():
    """Test evaluation metrics computation."""
    print("Testing evaluation metrics computation...")

    features = create_test_features()

    # Compute signatures and similarity matrix
    path_sig = PathSignature(order=2)
    signatures = path_sig.compute_signatures_dict(features)

    if not signatures:
        print("✗ No signatures computed for evaluation test")
        return

    softmax_model = SoftmaxRegression()
    similarity_matrix, song_names = softmax_model.compute_similarity_matrix(
        signatures, temperature=1.0
    )

    try:
        # Create synthetic ground truth
        ground_truth = {}
        for i, song in enumerate(song_names):
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3
            ground_truth[song] = set([song_names[j] for j in top_indices])

        # Evaluate performance
        evaluator = RecommendationMetrics()

        # Create recommendations from similarity matrix
        all_recommendations = []
        all_relevant_items = []
        for i, song in enumerate(song_names):
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3
            recommendations = [song_names[j] for j in top_indices]
            all_recommendations.append(recommendations)
            all_relevant_items.append(ground_truth[song])

        metrics = evaluator.compute_all_metrics(
            all_recommendations,
            all_relevant_items,
            similarity_matrix,
            song_names,
            k_values=[3],
        )

        print("✓ Evaluation metrics computed successfully")
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                if 3 in value:
                    print(f"  {metric_name}: {value[3]:.4f}")
                elif value:
                    first_val = next(iter(value.values()))
                    print(f"  {metric_name}: {first_val:.4f}")
                else:
                    print(f"  {metric_name}: (no data)")
            else:
                print(f"  {metric_name}: {float(value):.4f}")

    except Exception as e:
        print(f"✗ Evaluation metrics: Error - {str(e)}")

    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("ABLATION STUDIES TEST SUITE")
    print("=" * 60)
    print()

    # Run all tests
    test_path_signature_orders()
    test_temperature_scaling()
    test_feature_combinations()
    test_similarity_metrics()
    test_component_contributions()
    test_evaluation_metrics()

    print("=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
