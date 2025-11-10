# Benchmark and Ablation Study Results

This document summarizes the results of the initial benchmarks and ablation studies for the Belavkin Optimizer on the Modular Arithmetic task.

## Key Findings

1.  **Adam is the Strongest Baseline:** The Adam optimizer, when paired with an embedding layer in the model, was able to solve the Modular Arithmetic task perfectly, achieving 100% accuracy. This indicates that the task is learnable and that the model architecture is appropriate.

2.  **Belavkin Optimizer Underperforms:** The Belavkin Optimizer, in all its tested configurations, failed to learn the task. The accuracy remained close to zero, even with extensive hyperparameter tuning and an extended training period.

3.  **Ablation Studies Show No Improvement:** The ablation studies, which involved removing the damping and exploration components of the optimizer, did not lead to any improvement in performance. This suggests that the issue is not with a single component, but rather with the fundamental update rule of the optimizer.

4.  **Update Rule Correction was Ineffective:** An attempt to correct the update rule by flipping the sign of the damping term did not resolve the issue.

## Conclusion

The initial results of the Belavkin Optimizer are not promising. The optimizer is not competitive with Adam on the Modular Arithmetic task, and the core components of the optimizer do not seem to be contributing to learning.

Further research and experimentation are needed to understand the theoretical underpinnings of the optimizer and to develop a more effective update rule. It is possible that the current implementation is a misinterpretation of the Belavkin equation, or that the equation itself is not well-suited for this type of optimization problem.

## Raw Results

| config                                              | final_accuracy |
|-----------------------------------------------------|----------------|
| adam_lr0.01_gamma0.0001_beta0.01_embedding.json     | 1.000          |
| adam_lr0.01_gamma0.0001_beta0.01.json               | 0.165          |
| belavkin_lr0.01_gamma0.001_beta0.01.json            | 0.027          |
| belavkin_lr0.001_gamma0.0001_beta0.01.json         | 0.026          |
| sgd_lr0.01_gamma0.0001_beta0.01.json                | 0.020          |
| belavkin_lr0.01_gamma0.001_beta0.001.json           | 0.020          |
| belavkin_lr0.001_gamma0.01_beta0.01.json            | 0.015          |
| belavkin_lr0.001_gamma0.001_beta0.001.json         | 0.015          |
| belavkin_lr0.01_gamma0.001_beta0.1.json             | 0.014          |
| belavkin_lr0.0001_gamma0.0001_beta0.0_embedding.json| 0.013          |
| belavkin_lr0.001_gamma0.0001_beta0.001.json         | 0.013          |
| belavkin_lr0.001_gamma0.01_beta0.001.json           | 0.013          |
| belavkin_lr0.0001_gamma0.0001_beta0.1.json          | 0.012          |
| belavkin_lr0.01_gamma0.01_beta0.01.json             | 0.011          |
| belavkin_lr0.01_gamma0.0001_beta0.001.json          | 0.011          |
| belavkin_lr0.0001_gamma0.01_beta0.001.json          | 0.011          |
| belavkin_lr0.0001_gamma0.001_beta0.1.json           | 0.011          |
| belavkin_lr0.01_gamma0.0001_beta0.1.json            | 0.010          |
| belavkin_lr0.001_gamma0.001_beta0.01.json           | 0.010          |
| belavkin_lr0.001_gamma0.0001_beta0.1.json           | 0.010          |
| belavkin_lr0.0001_gamma0.0_beta0.001_embedding.json | 0.010          |
| belavkin_lr0.01_gamma0.01_beta0.1.json              | 0.010          |
| belavkin_lr0.01_gamma0.0001_beta0.01.json           | 0.010          |
| belavkin_lr0.0001_gamma0.01_beta0.01.json           | 0.009          |
| belavkin_lr0.0001_gamma0.001_beta0.01.json          | 0.008          |
| belavkin_lr0.001_gamma0.001_beta0.1.json            | 0.008          |
| belavkin_lr0.0001_gamma0.01_beta0.1.json            | 0.007          |
| belavkin_lr0.0001_gamma0.001_beta0.001.json         | 0.007          |
| belavkin_lr0.0001_gamma0.0001_beta0.001.json        | 0.007          |
| belavkin_lr0.01_gamma0.01_beta0.001.json            | 0.006          |
| belavkin_lr0.0001_gamma0.0001_beta0.01.json         | 0.005          |
| belavkin_lr0.0001_gamma0.0001_beta0.001_embedding.json| 0.005          |
| belavkin_lr0.001_gamma0.01_beta0.1.json             | 0.004          |
