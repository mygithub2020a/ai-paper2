"""
Benchmark suite for comparing Belavkin optimizer against baselines.
"""

from .trainer import train_model, evaluate_model
from .comparison import OptimizerComparison, run_benchmark

__all__ = ["train_model", "evaluate_model", "OptimizerComparison", "run_benchmark"]
