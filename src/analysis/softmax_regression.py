# pylint: disable=broad-except
# pylint: disable=invalid-name
# pylint: disable=line-too-long
"""
Softmax regression implementation for music categorisation and similarity computation.

This module provides a SoftmaxRegression class that uses path signatures to learn
musical categories and compute shape-based similarity between songs with configurable
temperature scaling and similarity weights.
"""

from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ..utils.logger_config import setup_logger
from ..utils.metadata import load_tracks_metadata

# Set up logger
logger = setup_logger("softmax_regression")


class SoftmaxRegression:
    """
    Softmax Regression model for music categorisation and similarity computation.

    This model uses path signatures to learn musical categories and compute
    shape-based similarity between songs. The similarity computation includes
    a sigmoid transformation for better contrast that can be configured.

    The sigmoid transformation in temperature scaling helps create sharper
    distinctions between similar and dissimilar items by applying:
    sigmoid(x) = 1 / (1 + exp(-steepness * (x - center)))

    The similarity computation combines three components with configurable weights:
    1. Path signature similarity (cosine similarity) - captures musical shape
    2. Category probability similarity - captures learned category patterns
    3. Category agreement (binary) - captures exact category matches

    Example usage:
        # Default configuration (moderate contrast)
        model = SoftmaxRegression(order=2)

        # High contrast configuration for clear genre boundaries
        model = SoftmaxRegression(
            sigmoid_steepness=10.0,  # Sharper transitions
            sigmoid_center=0.6       # Center around higher similarity
        )

        # Gentle contrast for subtle genre variations
        model = SoftmaxRegression(
            sigmoid_steepness=2.0,   # Softer transitions
            sigmoid_center=0.4       # Center around lower similarity
        )

        # Custom similarity weights (focus on path signatures)
        model = SoftmaxRegression(
            similarity_weights=[0.8, 0.15, 0.05]  # [path_sig, prob_sim, cat_agreement]
        )

        # Update parameters after initialisation
        model.set_sigmoid_parameters(sigmoid_steepness=8.0)
        model.set_similarity_weights([0.7, 0.2, 0.1])

        # Get current parameters
        sigmoid_params = model.get_sigmoid_parameters()
        weight_params = model.get_similarity_weights()
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iterations=1000,
        tolerance=1e-4,
        n_categories=5,
        use_genre_labels=True,
        similarity_weights=None,
        sigmoid_steepness=5.0,
        sigmoid_center=0.5,
    ):
        """
        Initialise the Softmax Regression model for music categorisation.

        Parameters:
        - learning_rate: Learning rate for gradient descent
        - max_iterations: Maximum number of iterations
        - tolerance: Convergence tolerance
        - n_categories: Number of music categories (default: 5)
        - use_genre_labels: Whether to use actual genre labels instead of clustering
        - similarity_weights: Weights for similarity components [path_sig, prob_sim, cat_agreement]
        - sigmoid_steepness: Steepness parameter for sigmoid transformation in temperature scaling (default: 5.0)
        - sigmoid_center: Center point for sigmoid transformation (default: 0.5)

        Note: The sigmoid transformation parameters significantly affect similarity scaling:
        - sigmoid_steepness: Higher values create sharper contrast between similar/dissimilar items
        - sigmoid_center: The similarity value around which the transformation is centered

        Note: similarity_weights must have exactly 3 elements:
        - similarity_weights[0]: Weight for path signature similarity (cosine similarity)
        - similarity_weights[1]: Weight for category probability similarity
        - similarity_weights[2]: Weight for category agreement (binary)
        Weights should sum to approximately 1.0 for proper normalisation.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_categories = n_categories
        self.use_genre_labels = use_genre_labels
        self.similarity_weights = similarity_weights or [
            0.7,
            0.2,
            0.1,
        ]  # Focus on path signatures

        # Validate similarity weights
        if len(self.similarity_weights) != 3:
            raise ValueError(
                "similarity_weights must have exactly 3 elements: [path_sig_weight, prob_sim_weight, cat_agreement_weight]"
            )

        # Validate weights sum to approximately 1.0
        weight_sum = sum(self.similarity_weights)
        if not 0.95 <= weight_sum <= 1.05:
            logger.warning(
                "Similarity weights sum to %.3f (should be close to 1.0)", weight_sum
            )

        self.sigmoid_steepness = sigmoid_steepness
        self.sigmoid_center = sigmoid_center
        self.weights = None
        self.bias = None
        self.genre_mapping = None
        self.similarity_matrix = None
        self.song_names = None

        # Validate sigmoid parameters
        if sigmoid_steepness <= 0:
            raise ValueError("sigmoid_steepness must be positive")
        if not 0 <= sigmoid_center <= 1:
            raise ValueError("sigmoid_center must be between 0 and 1")

        logger.info(
            "Initialised SoftmaxRegression model with %d categories, use_genre_labels=%s, "
            "sigmoid_steepness=%.1f, sigmoid_center=%.1f",
            n_categories,
            use_genre_labels,
            sigmoid_steepness,
            sigmoid_center,
        )

    def set_sigmoid_parameters(self, sigmoid_steepness=None, sigmoid_center=None):
        """
        Update the sigmoid transformation parameters for temperature scaling.

        Parameters:
        - sigmoid_steepness: New steepness parameter for sigmoid transformation
        - sigmoid_center: New center point for sigmoid transformation

        Note: Only the specified parameters will be updated. Others remain unchanged.
        """
        if sigmoid_steepness is not None:
            if sigmoid_steepness <= 0:
                raise ValueError("sigmoid_steepness must be positive")
            self.sigmoid_steepness = sigmoid_steepness

        if sigmoid_center is not None:
            if not 0 <= sigmoid_center <= 1:
                raise ValueError("sigmoid_center must be between 0 and 1")
            self.sigmoid_center = sigmoid_center

        logger.info(
            "Updated sigmoid parameters: steepness=%.1f, center=%.1f",
            self.sigmoid_steepness,
            self.sigmoid_center,
        )

    def get_sigmoid_parameters(self):
        """
        Get the current sigmoid transformation parameters.

        Returns:
        - dict: Dictionary containing 'sigmoid_steepness' and 'sigmoid_center' values
        """
        return {
            "sigmoid_steepness": self.sigmoid_steepness,
            "sigmoid_center": self.sigmoid_center,
        }

    def set_similarity_weights(self, similarity_weights):
        """
        Update the similarity weights for enhanced similarity computation.

        Parameters:
        - similarity_weights: List of exactly 3 weights [path_sig_weight, prob_sim_weight, cat_agreement_weight]

        Note: Weights should sum to approximately 1.0 for proper normalisation.
        """
        if len(similarity_weights) != 3:
            raise ValueError(
                "similarity_weights must have exactly 3 elements: [path_sig_weight, prob_sim_weight, cat_agreement_weight]"
            )

        weight_sum = sum(similarity_weights)
        if not 0.95 <= weight_sum <= 1.05:
            logger.warning(
                "Similarity weights sum to %.3f (should be close to 1.0)", weight_sum
            )

        self.similarity_weights = similarity_weights
        logger.info(
            "Updated similarity weights: path_sig=%.2f, prob_sim=%.2f, cat_agreement=%.2f",
            similarity_weights[0],
            similarity_weights[1],
            similarity_weights[2],
        )

    def get_similarity_weights(self):
        """
        Get the current similarity weights.

        Returns:
        - dict: Dictionary containing the three similarity weights
        """
        return {
            "path_sig_weight": self.similarity_weights[0],
            "prob_sim_weight": self.similarity_weights[1],
            "cat_agreement_weight": self.similarity_weights[2],
        }

    def fit(self, X, y):
        """
        Fit the softmax regression model.

        Parameters:
        - X: Path signature features of shape (n_samples, n_features)
        - y: Target category labels of shape (n_samples,) with values 0-4
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional matrix")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be one-dimensional and aligned with X")
        if not np.issubdtype(y.dtype, np.integer):
            if not np.isfinite(y.astype(np.float64)).all() or not np.equal(
                y, np.floor(y.astype(np.float64))
            ).all():
                raise ValueError("y must contain integer category labels")
        y = y.astype(int)
        if np.any(y < 0) or np.any(y >= self.n_categories):
            raise ValueError("y contains a category outside the configured range")

        n_samples, n_features = X.shape

        # Initialise weights and bias
        # Retain the established MT19937 initialisation sequence without
        # mutating NumPy's process-global random state.
        local_random = np.random.RandomState(2025)
        self.weights = local_random.randn(n_features, self.n_categories) * 0.01
        self.bias = np.zeros(self.n_categories)

        # Convert labels to one-hot encoding
        y_onehot = np.zeros((n_samples, self.n_categories))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1

        # Gradient descent
        prev_loss = float("inf")
        for iteration in range(self.max_iterations):
            # Forward pass
            scores = np.dot(X, self.weights) + self.bias
            probs = self._softmax(scores)

            # Compute cross-entropy loss
            loss = -np.sum(y_onehot * np.log(probs + 1e-10)) / n_samples

            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                logger.info("Converged after %d iterations", iteration + 1)
                break

            prev_loss = loss

            # Backward pass
            gradient = probs - y_onehot
            self.weights -= self.learning_rate * np.dot(X.T, gradient) / n_samples
            self.bias -= self.learning_rate * np.sum(gradient, axis=0) / n_samples

            if (iteration + 1) % 100 == 0:
                logger.info("Iteration %d, Loss: %.4f", iteration + 1, loss)

    def _softmax(self, scores):
        """Compute softmax probabilities."""
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict_proba(self, X):
        """
        Predict category probabilities.

        Parameters:
        - X: Path signature features of shape (n_samples, n_features)

        Returns:
        - probabilities: Category probabilities of shape (n_samples, n_categories)
        """
        scores = np.dot(X, self.weights) + self.bias
        return self._softmax(scores)

    def predict(self, X):
        """
        Predict category labels.

        Parameters:
        - X: Path signature features of shape (n_samples, n_features)

        Returns:
        - predictions: Category labels of shape (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _convert_signatures_to_matrix(self, signatures_dict):
        """
        Convert signatures dictionary to matrix format.

        Parameters:
        - signatures_dict: Dictionary mapping song names to path signatures

        Returns:
        - X: Signature matrix of shape (n_songs, n_features)
        - song_names: List of song names
        """
        if not signatures_dict:
            raise ValueError("signatures must be a non-empty mapping")

        song_names = list(signatures_dict.keys())
        signatures = []
        signature_length = None
        for song_name, raw_signature in signatures_dict.items():
            try:
                signature = np.asarray(raw_signature, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"signature for {song_name} must be numeric") from error
            if signature.ndim != 1 or signature.size == 0:
                raise ValueError(
                    f"signature for {song_name} must be a non-empty one-dimensional vector"
                )
            if not np.isfinite(signature).all():
                raise ValueError(f"signature for {song_name} must be finite")
            if signature_length is None:
                signature_length = int(signature.size)
            elif signature.size != signature_length:
                raise ValueError("all signatures must have equal length")
            signatures.append(signature)

        X = np.stack(signatures, axis=0)
        logger.info(
            "Converted %d signatures to matrix of shape %s", len(signatures), X.shape
        )

        return X, song_names

    def _create_music_specific_categories(self, X, tracks_json=None):
        """
        Create music-specific categories using actual genre information.

        Parameters:
        - X: Signature matrix of shape (n_songs, n_features)
        - tracks_json: Path to tracks metadata JSON file

        Returns:
        - y: Category labels for each song
        """
        if self.use_genre_labels and tracks_json:
            try:
                # Load genre information from metadata
                _, _, tracks = load_tracks_metadata(tracks_json)

                # Extract genres and create mapping
                genres = []
                for track in tracks:
                    genre = track.get("genre", "Unknown")
                    if genre not in genres:
                        genres.append(genre)

                # Create genre to category mapping
                self.genre_mapping = {
                    genre: i for i, genre in enumerate(genres[: self.n_categories])
                }

                # Assign categories based on genres
                y = []
                for track in tracks:
                    genre = track.get("genre", "Unknown")
                    if genre in self.genre_mapping:
                        y.append(self.genre_mapping[genre])
                    else:
                        y.append(0)  # Default category

                y = np.array(y)

                # Log genre distribution
                unique_labels, counts = np.unique(y, return_counts=True)
                genre_counts = {
                    genres[i] if i < len(genres) else f"Category_{i}": count
                    for i, count in zip(unique_labels, counts)
                }
                logger.info("Genre-based categories: %s", genre_counts)

                return y

            except Exception as e:
                logger.warning(
                    "Failed to use genre labels: %s. Falling back to clustering.",
                    str(e),
                )

        # Fallback to improved clustering
        return self._create_improved_clustering(X)

    def _create_improved_clustering(self, X):
        """
        Create improved clustering using music-specific features.

        Parameters:
        - X: Signature matrix of shape (n_songs, n_features)

        Returns:
        - y: Category labels for each song
        """

        # Standardise features for better clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use PCA to reduce dimensionality for better clustering
        if X_scaled.shape[1] > 50:  # Only if we have many features
            # PCA can't have more components than samples
            max_components = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1] - 1)
            pca = PCA(n_components=max_components, random_state=2025)
            X_scaled = pca.fit_transform(X_scaled)
            logger.info("Applied PCA reduction to %d components", X_scaled.shape[1])

        # Use more robust clustering parameters
        kmeans = KMeans(
            n_clusters=self.n_categories,
            random_state=2025,
            n_init=20,  # More initialisations for better results
            max_iter=500,
        )
        y = kmeans.fit_predict(X_scaled)

        # Log clustering results
        unique_labels, counts = np.unique(y, return_counts=True)
        logger.info("Clustering results: %s", dict(zip(unique_labels, counts)))

        return y

    def _compute_enhanced_similarity(self, sig1, sig2, prob1, prob2):
        """
        Compute enhanced similarity between two songs with focus on path signatures.

        Parameters:
        - sig1, sig2: Path signatures of the two songs
        - prob1, prob2: Category probability distributions of the two songs

        Returns:
        - similarity: Enhanced similarity score
        """
        # Method 1: Path signature similarity (cosine similarity) - PRIMARY FOCUS
        # This captures how similar the musical SHAPES are
        sig_sim = np.dot(sig1, sig2) / (
            np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-8
        )

        # Apply non-linear transformation to enhance differences
        sig_sim = np.power(sig_sim, 2)  # Square to emphasise higher similarities

        # Method 2: Category probability similarity (secondary)
        prob_sim = np.dot(prob1, prob2) / (
            np.linalg.norm(prob1) * np.linalg.norm(prob2) + 1e-8
        )

        # Method 3: Category agreement (tertiary)
        cat1 = np.argmax(prob1)
        cat2 = np.argmax(prob2)
        cat_agreement = 1.0 if cat1 == cat2 else 0.0

        # Combine methods with emphasis on path signature similarity
        similarity = (
            self.similarity_weights[0] * sig_sim
            + self.similarity_weights[1] * prob_sim
            + self.similarity_weights[2] * cat_agreement
        )

        return similarity

    def _compute_enhanced_similarity_matrix(self, signatures, probabilities):
        """Vectorise the established pairwise composite-similarity formula."""

        signatures = np.asarray(signatures, dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if signatures.ndim != 2 or probabilities.ndim != 2:
            raise ValueError("signatures and probabilities must be two-dimensional")
        if signatures.shape[0] != probabilities.shape[0] or signatures.shape[0] == 0:
            raise ValueError("signatures and probabilities must have aligned rows")
        if not np.isfinite(signatures).all() or not np.isfinite(probabilities).all():
            raise ValueError("signatures and probabilities must be finite")

        signature_dot = signatures @ signatures.T
        signature_norm = np.linalg.norm(signatures, axis=1)
        signature_cosine = signature_dot / (
            np.outer(signature_norm, signature_norm) + 1e-8
        )
        signature_component = np.square(signature_cosine)

        probability_dot = probabilities @ probabilities.T
        probability_norm = np.linalg.norm(probabilities, axis=1)
        probability_component = probability_dot / (
            np.outer(probability_norm, probability_norm) + 1e-8
        )
        categories = np.argmax(probabilities, axis=1)
        category_component = (categories[:, None] == categories[None, :]).astype(
            np.float64
        )

        similarity = (
            self.similarity_weights[0] * signature_component
            + self.similarity_weights[1] * probability_component
            + self.similarity_weights[2] * category_component
        )
        np.fill_diagonal(similarity, 1.0)
        if not np.isfinite(similarity).all():
            raise ValueError("composite similarity must be finite")
        return similarity

    def _apply_temperature_scaling(self, similarity_matrix, temperature):
        """
        Apply improved temperature scaling to make differences more pronounced.

        The scaling process involves:
        1. Power transformation using inverse temperature
        2. Sigmoid transformation for better contrast using configurable parameters
        3. Renormalisation to [0,1] range

        Parameters:
        - similarity_matrix: Raw similarity matrix
        - temperature: Temperature parameter (lower = more pronounced differences)

        Returns:
        - scaled_matrix: Temperature-scaled similarity matrix

        Note: The sigmoid transformation uses self.sigmoid_steepness and self.sigmoid_center
        parameters to control the contrast enhancement. Higher steepness values create
        sharper transitions between similar and dissimilar items.
        """
        if (
            isinstance(temperature, bool)
            or not np.isscalar(temperature)
            or not np.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise ValueError("temperature must be a positive finite number")
        similarity_matrix = np.asarray(similarity_matrix, dtype=np.float64)
        if similarity_matrix.ndim != 2 or not np.isfinite(similarity_matrix).all():
            raise ValueError("similarity_matrix must be a finite two-dimensional array")
        if np.any(similarity_matrix < 0.0):
            raise ValueError("temperature scaling requires non-negative similarities")
        if float(temperature) == 1.0:
            return similarity_matrix.copy()

        # Use inverse temperature to make differences more pronounced
        scaled_matrix = np.power(similarity_matrix, 1 / float(temperature))

        # Apply sigmoid-like transformation for better contrast
        scaled_matrix = 1 / (
            1 + np.exp(-self.sigmoid_steepness * (scaled_matrix - self.sigmoid_center))
        )

        # Renormalise to [0,1] range
        scaled_matrix = (scaled_matrix - np.min(scaled_matrix)) / (
            np.max(scaled_matrix) - np.min(scaled_matrix) + 1e-8
        )

        return scaled_matrix

    def compute_similarity_matrix(
        self, signatures_dict, temperature=1.0, tracks_json=None
    ):
        """
        Compute enhanced similarity matrix using improved path signature analysis.

        Path signatures capture the SHAPE of musical progression, not magnitude.
        We use them to learn musical categories and compute shape-based similarity.

        Parameters:
        - signatures_dict: Dictionary mapping song names to path signatures
        - temperature: Temperature parameter for softmax
        - tracks_json: Path to tracks metadata for genre-based categories

        Returns:
        - similarity_matrix: Matrix of similarity scores
        - song_names: List of song names
        """
        if not signatures_dict:
            raise ValueError("signatures must be a non-empty mapping")

        # Step 1: Convert signatures to matrix format
        X, song_names = self._convert_signatures_to_matrix(signatures_dict)

        n_songs = len(song_names)

        # Step 2: Create music-specific categories
        y = self._create_music_specific_categories(X, tracks_json)

        # Step 3: Train softmax regression model on path signatures
        logger.info("Training softmax regression model...")
        self.fit(X, y)

        # Step 4: Get category probabilities for all songs
        category_probs = self.predict_proba(X)
        logger.info(
            "Category probability ranges - min: %.4f, max: %.4f",
            np.min(category_probs),
            np.max(category_probs),
        )

        # Step 5: Compute enhanced similarity matrix
        similarity_matrix = self._compute_enhanced_similarity_matrix(
            X, category_probs
        )

        # Step 6: Apply temperature scaling
        similarity_matrix = self._apply_temperature_scaling(
            similarity_matrix, temperature
        )

        # Store for recommendations
        self.similarity_matrix = similarity_matrix
        self.song_names = song_names

        logger.info(
            "Computed similarity matrix with shape %s, range [%.4f, %.4f]",
            similarity_matrix.shape,
            np.min(similarity_matrix),
            np.max(similarity_matrix),
        )

        return similarity_matrix, song_names

    def recommend(
        self, user_id: str, n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Parameters:
        - user_id: User identifier (song name in this case)
        - n_recommendations: Number of recommendations to generate

        Returns:
        - recommendations: List of (song_name, similarity_score) tuples
        """
        if not hasattr(self, "similarity_matrix") or not hasattr(self, "song_names"):
            logger.error("Model not properly initialised with similarity matrix")
            return []

        try:
            # Find the index of the user's song
            if user_id not in self.song_names:
                logger.warning(
                    "Path Signature: User ID '%s' not found in song names. "
                    "Available: %d songs. First few: %s",
                    user_id,
                    len(self.song_names),
                    self.song_names[:5] if len(self.song_names) > 0 else "none",
                )
                # Try to find closest match (for debugging)
                if len(self.song_names) > 0:
                    # Return top items by average similarity as fallback
                    avg_similarities = np.mean(self.similarity_matrix, axis=0)
                    top_indices = np.argsort(avg_similarities)[-n_recommendations:][
                        ::-1
                    ]
                    return [
                        (self.song_names[idx], float(avg_similarities[idx]))
                        for idx in top_indices
                    ]
                return []

            user_index = self.song_names.index(user_id)

            # Get similarities for this user
            similarities = self.similarity_matrix[user_index].copy()
            similarities[user_index] = -1  # Exclude self

            # Get top recommendations
            top_indices = np.argsort(similarities)[-n_recommendations:][::-1]

            recommendations = []
            for idx in top_indices:
                # Include all similarities (positive or negative) as cosine similarity ranges from -1 to 1
                # Filter only if similarity is exactly -1 (self-similarity which we set)
                if similarities[idx] > -0.99:  # Allow slightly negative similarities
                    recommendations.append(
                        (self.song_names[idx], float(similarities[idx]))
                    )

            # If we don't have enough recommendations, return what we have
            if len(recommendations) < n_recommendations:
                logger.warning(
                    "Path Signature: Only found %d recommendations (requested %d) for %s. "
                    "Similarity range: [%.4f, %.4f]",
                    len(recommendations),
                    n_recommendations,
                    user_id,
                    np.min(similarities),
                    np.max(similarities),
                )

            return recommendations[:n_recommendations]

        except ValueError as ve:
            logger.error("Path Signature: ValueError for user ID %s: %s", user_id, ve)
            return []
        except Exception as e:
            logger.error(
                "Path Signature: Error generating recommendations for %s: %s",
                user_id,
                str(e),
                exc_info=True,
            )
            return []
