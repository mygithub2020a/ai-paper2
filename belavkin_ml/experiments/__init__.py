"""
Experimental infrastructure for Belavkin optimizer evaluation.

Includes:
- Benchmark suite for optimizer comparison
- Training loops and evaluation
- Metrics collection and visualization
- Ablation study framework
"""

from belavkin_ml.experiments.benchmark import OptimizerBenchmark
from belavkin_ml.experiments.trainer import Trainer, TrainerConfig

__all__ = ["OptimizerBenchmark", "Trainer", "TrainerConfig"]
