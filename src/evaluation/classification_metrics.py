# pylint: disable=broad-except
# pylint: disable=invalid-name
"""
Classification metrics for evaluating music genre classification performance.

This module provides comprehensive evaluation metrics for classification tasks,
including both basic metrics and advanced statistical analysis capabilities.

Metrics Included:
    - Accuracy, Precision, Recall, F1-score: Basic classification metrics
    - Confusion Matrix: Detailed classification results visualisation
    - ROC AUC: Area under ROC curve for multi-class classification
    - Cross-validation: Robust performance assessment
    - Statistical Significance Testing: Paired t-tests for model comparison

Example:
    >>> from src.evaluation import ClassificationMetrics
    >>> metrics = ClassificationMetrics()
    >>> results = metrics.compute_all_metrics(y_true, y_pred, class_names=genres)
    >>> metrics.plot_confusion_matrix(y_true, y_pred, genres, "confusion.png")
    >>> metrics.print_metrics_summary(results)
"""

from typing import List, Dict, Optional, Union
import json
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.logger_config import setup_logger

logger = setup_logger("classification_metrics")


class ClassificationMetrics:
    """
    Comprehensive evaluation metrics for classification tasks.

    Implements standard classification evaluation metrics including:
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix analysis
    - Cross-validation support
    - Statistical significance testing
    """

    def __init__(self):
        """Initialise the ClassificationMetrics class."""
        self.metrics_history = {}

    def compute_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Compute basic classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class metrics ('micro', 'macro', 'weighted')

        Returns:
            Dictionary containing accuracy, precision, recall, and F1-score
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        # Add per-class metrics if multi-class
        if len(np.unique(y_true)) > 2:
            metrics["precision_macro"] = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics["recall_macro"] = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics["f1_macro"] = f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )

            metrics["precision_micro"] = precision_score(
                y_true, y_pred, average="micro", zero_division=0
            )
            metrics["recall_micro"] = recall_score(
                y_true, y_pred, average="micro", zero_division=0
            )
            metrics["f1_micro"] = f1_score(
                y_true, y_pred, average="micro", zero_division=0
            )

        return metrics

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix",
    ) -> None:
        """
        Plot confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save the plot
            title: Plot title
        """
        # Get unique classes that actually appear in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))

        # Compute confusion matrix with labels parameter to ensure consistent ordering
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

        # Filter class_names to only include classes that appear in the data
        if class_names is not None:
            filtered_class_names = [
                class_names[i] for i in unique_classes if i < len(class_names)
            ]
        else:
            filtered_class_names = None

        # Ensure proper color scaling - use raw counts with explicit vmin/vmax
        cm_max = cm.max() if cm.max() > 0 else 1
        cm_min = cm.min()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=filtered_class_names,
            yticklabels=filtered_class_names,
            vmin=cm_min,
            vmax=cm_max,
            square=False,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved confusion matrix to %s", save_path)

        plt.show()

    def compute_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes

        Returns:
            Formatted classification report string
        """
        # Get unique classes that actually appear in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))

        # Filter target_names to only include classes that appear in the data
        if target_names is not None:
            # Only include target names for classes that appear in the data
            # target_names[i] corresponds to class label i
            filtered_target_names = [
                target_names[i] for i in unique_classes if i < len(target_names)
            ]
            # Use labels parameter to specify which classes to include
            return classification_report(
                y_true,
                y_pred,
                labels=unique_classes,
                target_names=filtered_target_names,
                zero_division=0,
            )
        else:
            return classification_report(
                y_true, y_pred, labels=unique_classes, zero_division=0
            )

    def cross_validate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = "accuracy",
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.

        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for cross-validation

        Returns:
            Dictionary containing cross-validation results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=2025)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results = {
            f"{scoring}_mean": cv_scores.mean(),
            f"{scoring}_std": cv_scores.std(),
            f"{scoring}_scores": cv_scores.tolist(),
        }

        logger.info(
            "Cross-validation %s: %.3f (+/- %.3f)",
            scoring,
            results[f"{scoring}_mean"],
            2 * results[f"{scoring}_std"],
        )

        return results

    def compute_roc_auc(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, average: str = "weighted"
    ) -> float:
        """
        Compute ROC AUC score for multi-class classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            average: Averaging method ('micro', 'macro', 'weighted')

        Returns:
            ROC AUC score
        """
        try:
            return roc_auc_score(
                y_true, y_pred_proba, average=average, multi_class="ovr"
            )
        except Exception as e:
            logger.warning("Could not compute ROC AUC: %s", e)
            return 0.0

    def statistical_significance_test(
        self, scores1: List[float], scores2: List[float], alpha: float = 0.05
    ) -> Dict[str, Union[float, bool]]:
        """
        Perform statistical significance test between two sets of scores.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            alpha: Significance level

        Returns:
            Dictionary containing test statistic, p-value, and significance
        """
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        is_significant = p_value < alpha

        results = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "alpha": alpha,
        }

        logger.info(
            "Statistical test: t=%.3f, p=%.4f, significant=%s",
            t_stat,
            p_value,
            is_significant,
        )

        return results

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, str]]:
        """
        Compute all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            class_names: Names of classes

        Returns:
            Dictionary containing all metrics
        """
        metrics = {}

        # Basic metrics
        basic_metrics = self.compute_basic_metrics(y_true, y_pred)
        metrics.update(basic_metrics)

        # Confusion matrix
        metrics["confusion_matrix"] = self.compute_confusion_matrix(y_true, y_pred)

        # Classification report
        metrics["classification_report"] = self.compute_classification_report(
            y_true, y_pred, class_names
        )

        # ROC AUC if probabilities are available
        if y_pred_proba is not None:
            metrics["roc_auc_weighted"] = self.compute_roc_auc(
                y_true, y_pred_proba, "weighted"
            )
            metrics["roc_auc_macro"] = self.compute_roc_auc(
                y_true, y_pred_proba, "macro"
            )
            metrics["roc_auc_micro"] = self.compute_roc_auc(
                y_true, y_pred_proba, "micro"
            )

        # Store metrics history
        self.metrics_history = metrics

        logger.info("Computed all classification metrics")
        return metrics

    def print_metrics_summary(self, metrics: Dict[str, Union[float, str]]):
        """
        Print a formatted summary of classification metrics.

        Args:
            metrics: Dictionary containing all metrics
        """
        print("\n" + "=" * 60)
        print("CLASSIFICATION METRICS SUMMARY")
        print("=" * 60)

        # Print basic metrics
        print("Basic Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1_score', 0.0):.4f}")

        # Print macro metrics if available
        if "precision_macro" in metrics:
            print("\nMacro Metrics:")
            print(f"  Precision: {metrics['precision_macro']:.4f}")
            print(f"  Recall:    {metrics['recall_macro']:.4f}")
            print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

        # Print ROC AUC if available
        if "roc_auc_weighted" in metrics:
            print("\nROC AUC:")
            print(f"  Weighted: {metrics['roc_auc_weighted']:.4f}")
            print(f"  Macro:    {metrics['roc_auc_macro']:.4f}")
            print(f"  Micro:    {metrics['roc_auc_micro']:.4f}")

        # Print classification report
        if "classification_report" in metrics:
            print("\nDetailed Classification Report:")
            print(metrics["classification_report"])

        print("=" * 60)

    def save_metrics(self, metrics: Dict[str, Union[float, str]], output_file: str):
        """
        Save metrics to a JSON file.

        Args:
            metrics: Dictionary containing all metrics
            output_file: Path to output file
        """

        # Convert numpy types and remove non-serialisable objects
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, str):
                return obj
            else:
                return str(obj)

        # Convert metrics to JSON-serialisable format
        json_metrics = {}
        for key, value in metrics.items():
            if key not in ["confusion_matrix", "classification_report"]:
                json_metrics[key] = convert_numpy(value)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_metrics, f, indent=2)
            logger.info("Saved classification metrics to %s", output_file)
        except Exception as e:
            logger.error("Error saving metrics to %s: %s", output_file, e)

    def compare_models(
        self, model_results: Dict[str, Dict[str, float]], metric: str = "accuracy"
    ) -> Dict[str, float]:
        """
        Compare multiple models based on a specific metric.

        Args:
            model_results: Dictionary mapping model names to their metrics
            metric: Metric to compare on

        Returns:
            Dictionary mapping model names to their metric values
        """
        comparison = {}
        for model_name, results in model_results.items():
            if metric in results:
                comparison[model_name] = results[metric]

        # Sort by metric value (descending)
        sorted_comparison = dict(
            sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        )

        print(f"\nModel Comparison ({metric}):")
        for i, (model_name, value) in enumerate(sorted_comparison.items(), 1):
            print(f"  {i}. {model_name}: {value:.4f}")

        return sorted_comparison
