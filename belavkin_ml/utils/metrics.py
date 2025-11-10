"""
Metrics computation and aggregation utilities.
"""

import numpy as np
from typing import Dict, List, Any, Optional


def compute_convergence_metrics(
    train_accs: List[float],
    test_accs: List[float],
    target_accuracies: List[float] = [0.90, 0.95, 0.99],
) -> Dict[str, Any]:
    """
    Compute convergence-related metrics from training history.

    Args:
        train_accs: List of training accuracies per epoch
        test_accs: List of test accuracies per epoch
        target_accuracies: Target accuracy thresholds

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Best and final performance
    metrics['best_test_acc'] = max(test_accs)
    metrics['final_test_acc'] = test_accs[-1]
    metrics['best_train_acc'] = max(train_accs)
    metrics['final_train_acc'] = train_accs[-1]

    # Generalization gap
    metrics['train_test_gap'] = train_accs[-1] - test_accs[-1]
    metrics['best_gap'] = train_accs[np.argmax(test_accs)] - max(test_accs)

    # Epochs to targets
    metrics['epochs_to_target'] = {}
    for target in target_accuracies:
        for epoch, acc in enumerate(test_accs):
            if acc >= target:
                metrics['epochs_to_target'][target] = epoch
                break
        else:
            metrics['epochs_to_target'][target] = None

    # Convergence stability
    if len(test_accs) >= 10:
        last_10_accs = test_accs[-10:]
        metrics['final_stability'] = np.std(last_10_accs)
    else:
        metrics['final_stability'] = None

    # Learning speed (slope of accuracy in first 25% of training)
    n_early = max(1, len(test_accs) // 4)
    if n_early > 1:
        early_accs = test_accs[:n_early]
        metrics['early_learning_rate'] = (early_accs[-1] - early_accs[0]) / n_early
    else:
        metrics['early_learning_rate'] = None

    return metrics


def aggregate_results(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple seeds.

    Args:
        results: List of result dictionaries
        metrics: List of metrics to aggregate (default: all numerical metrics)

    Returns:
        Dictionary mapping metric names to {mean, std, min, max}
    """
    if not results:
        return {}

    # Auto-detect numerical metrics if not specified
    if metrics is None:
        metrics = []
        for key, value in results[0].items():
            if isinstance(value, (int, float, np.number)):
                metrics.append(key)

    aggregated = {}

    for metric in metrics:
        values = [r[metric] for r in results if metric in r and r[metric] is not None]

        if values:
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'n': len(values),
            }

    return aggregated


def compute_statistical_significance(
    results_a: List[Dict],
    results_b: List[Dict],
    metric: str = 'best_test_acc',
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute statistical significance between two sets of results.

    Uses Welch's t-test for unequal variances.

    Args:
        results_a: Results for method A
        results_b: Results for method B
        metric: Metric to compare
        alpha: Significance level

    Returns:
        Dictionary with test statistics
    """
    from scipy import stats

    values_a = [r[metric] for r in results_a if metric in r]
    values_b = [r[metric] for r in results_b if metric in r]

    if not values_a or not values_b:
        return {'error': 'Insufficient data'}

    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

    # Effect size (Cohen's d)
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    pooled_std = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    return {
        'mean_a': mean_a,
        'mean_b': mean_b,
        'std_a': np.std(values_a),
        'std_b': np.std(values_b),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': cohens_d,
        'effect_size': _interpret_cohens_d(abs(cohens_d)),
    }


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def compute_sample_efficiency(
    results: Dict[str, List[Dict]],
    target_acc: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute sample efficiency: examples needed to reach target accuracy.

    Args:
        results: Dictionary mapping optimizer names to results
        target_acc: Target accuracy threshold

    Returns:
        Dictionary mapping optimizer names to sample statistics
    """
    efficiency = {}

    for opt_name, opt_results in results.items():
        samples_to_target = []

        for result in opt_results:
            steps = result['steps_to_target'].get(target_acc)
            if steps is not None:
                # Assuming we know batch size from result
                batch_size = result.get('batch_size', 32)
                samples = steps * batch_size
                samples_to_target.append(samples)

        if samples_to_target:
            efficiency[opt_name] = {
                'mean_samples': np.mean(samples_to_target),
                'std_samples': np.std(samples_to_target),
                'min_samples': np.min(samples_to_target),
                'success_rate': len(samples_to_target) / len(opt_results),
            }
        else:
            efficiency[opt_name] = {
                'mean_samples': None,
                'success_rate': 0.0,
            }

    return efficiency


def create_summary_table(
    results: Dict[str, List[Dict]],
    metrics: List[str] = ['best_test_acc', 'total_time', 'train_test_gap'],
) -> str:
    """
    Create a markdown table summarizing results.

    Args:
        results: Dictionary mapping optimizer names to results
        metrics: Metrics to include in table

    Returns:
        Markdown-formatted table string
    """
    # Aggregate results
    agg_results = {}
    for opt_name, opt_results in results.items():
        # Find best hyperparameter configuration
        configs = {}
        for result in opt_results:
            key = (result['lr'], result.get('gamma'), result.get('beta'))
            if key not in configs:
                configs[key] = []
            configs[key].append(result)

        # Best config by test accuracy
        best_config = max(configs.items(), key=lambda x: np.mean([r['best_test_acc'] for r in x[1]]))
        agg_results[opt_name] = aggregate_results(best_config[1], metrics)

    # Create table
    table = "| Optimizer | " + " | ".join([m.replace('_', ' ').title() for m in metrics]) + " |\n"
    table += "|" + "---|" * (len(metrics) + 1) + "\n"

    for opt_name, agg in agg_results.items():
        row = f"| {opt_name} |"
        for metric in metrics:
            if metric in agg:
                mean = agg[metric]['mean']
                std = agg[metric]['std']
                row += f" {mean:.4f} ± {std:.4f} |"
            else:
                row += " N/A |"
        table += row + "\n"

    return table
