"""PyImgAno - Enterprise-Grade Visual Anomaly Detection Toolkit."""

from . import datasets, models, preprocessing, utils, visualization
from .benchmark import AlgorithmBenchmark, quick_benchmark
from .evaluation import (
    compute_auroc,
    compute_average_precision,
    compute_classification_metrics,
    evaluate_detector,
    find_optimal_threshold,
    print_evaluation_summary,
)

__version__ = "0.2.0"

__all__ = [
    # Modules
    "datasets",
    "models",
    "preprocessing",
    "utils",
    "visualization",
    # Evaluation
    "evaluate_detector",
    "compute_auroc",
    "compute_average_precision",
    "compute_classification_metrics",
    "find_optimal_threshold",
    "print_evaluation_summary",
    # Benchmark
    "AlgorithmBenchmark",
    "quick_benchmark",
]
