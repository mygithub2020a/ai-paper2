# Belavkin Optimization Framework

This project is an implementation of the Belavkin Optimization Framework for classical and quantum machine learning, as detailed in the provided research brief.

This repository currently contains the initial implementation of the Belavkin Optimizer in PyTorch and a set of synthetic benchmarks to evaluate its performance against standard optimizers.

## Belavkin Optimizer

The optimizer is implemented in `optimizer/belavkin_optimizer.py` and follows the update rule:

`dθ = -[γ * ∇L(θ) + η * (∇L(θ) - m)^2] + β * |∇L(θ) - m| * ε`

where:
*   `γ`: Adaptive learning rate parameter
*   `η`: Non-linear collapse strength parameter
*   `m`: Exponential moving average of past gradients
*   `β`: Stochastic exploration factor
*   `ε ~ N(0,1)`: Standard Gaussian noise

## Initial Benchmark Results

The initial benchmarks were performed on three synthetic tasks:

1.  **Non-Markovian Quadratic Optimization:** A quadratic optimization problem where the loss at the current step depends on the history of the parameters.
2.  **Modular Arithmetic (Addition):** A simple task to learn a modular addition operation.
3.  **Multi-Objective Quadratic Problem:** A quadratic optimization problem with multiple conflicting objectives.

The results of these benchmarks can be found in the `benchmarks/synthetic_benchmarks.ipynb` Jupyter notebook. The notebook contains the code to reproduce the experiments and visualizations of the convergence of the Belavkin Optimizer compared to Adam and SGD.

A summary of the initial findings:

*   **Non-Markovian Quadratic:** The Belavkin optimizer demonstrates competitive performance, converging at a similar rate to Adam.
*   **Modular Addition:** All optimizers solve this simple task quickly.
*   **Multi-Objective Quadratic:** The Belavkin optimizer shows a stable convergence profile, comparable to Adam.

These are preliminary results on small-scale synthetic problems. Further benchmarking on more complex and larger-scale tasks is required to draw more definitive conclusions about the optimizer's properties and practical advantages.

## How to Run the Benchmarks

1.  Install the required dependencies:
    ```bash
    pip install torch jupyter matplotlib
    ```
2.  Run the Jupyter notebook:
    ```bash
    jupyter notebook benchmarks/synthetic_benchmarks.ipynb
    ```
