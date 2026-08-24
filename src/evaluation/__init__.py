"""Evaluation package with lazy legacy exports.

The canonical task-contract submodule must remain importable without loading
classification, cross-validation, plotting, or other scientific dependencies.
Legacy public classes are retained lazily for compatibility until their repair
units decide the final API.
"""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "RecommendationMetrics": (".recommendation_metrics", "RecommendationMetrics"),
    "ClassificationMetrics": (".classification_metrics", "ClassificationMetrics"),
    "CrossValidator": (".cross_validation", "CrossValidator"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a legacy public class only when that exact export is requested."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_EXPORTS))
