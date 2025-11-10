# Belavkin ML: Experimental Results Summary

**Date**: November 10, 2025
**Experiments Run**: Track 1 Optimizer Benchmarks, Ablation Studies, Sparse Parity
**Status**: Initial results collected

---

## Executive Summary

This report summarizes initial experimental results for the Belavkin Quantum Filtering ML research project. We have successfully executed multiple experiments across both Track 1 (Neural Network Optimization) and partial Track 2 (Reinforcement Learning).

### Key Findings

1. **Sparse Parity Task**: Adam significantly outperforms Belavkin optimizer
   - Adam: 100% test accuracy (perfect learning)
   - Belavkin: 54.5% test accuracy
   - **Conclusion**: Belavkin struggles with sparse structure discovery

2. **Modular Arithmetic (p=13, preliminary)**: All optimizers struggled with limited training
   - Best: Adam with 29.41% accuracy
   - SGD: 18.82% accuracy
   - Belavkin: 17.65% accuracy
   - **Note**: Task requires more epochs; 50 epochs insufficient

3. **Modular Arithmetic (p=97)**: Experiments running (results pending)

4. **Ablation Study**: In progress, testing component contributions

---

## Experiment 1: Sparse Parity (n=10 bits, k=3 sparse)

### Setup
- **Task**: Learn 3-sparse XOR function from 10 input bits
- **Dataset**: 1000 examples (800 train, 200 test)
- **Relevant bits**: [8, 1, 5]
- **Model**: 2-layer MLP with [128, 128] hidden units
- **Training**: 150 epochs max, batch size 64

### Results

| Optimizer | Test Accuracy | Learning Rate | Gamma | Beta |
|-----------|--------------|---------------|-------|------|
| **Adam**  | **100.0%**   | 1e-3          | -     | -    |
| Belavkin  | 54.5%        | 1e-3          | 1e-4  | 1e-2 |

### Analysis

**Adam's Success**:
- Achieved perfect generalization
- Discovered the sparse structure
- Standard adaptive learning rate was sufficient

**Belavkin's Limitation**:
- Only slightly better than random (50%)
- Gradient-dependent damping may have hindered exploration
- Multiplicative noise didn't help with discrete structure

**Visualization**: Learning curves, loss curves, and comparison plots generated in:
`experiments/track1_optimizer/benchmarks/sparse_parity_n10_k3/`

---

## Experiment 2: Modular Arithmetic - Preliminary (p=13)

### Setup
- **Task**: f(x,y) = (x + y) mod 13
- **Dataset**: 169 examples (84 train, 85 test)
- **Model**: 2-layer MLP with [128, 128] hidden units
- **Training**: 50 epochs, batch size 64

### Results

| Optimizer | Best Test Acc | Mean ± Std (2 seeds) | Learning Rate |
|-----------|---------------|---------------------|---------------|
| Adam      | 29.41%        | 24.71% ± 4.71%      | 3e-3          |
| SGD       | 18.82%        | 18.24% ± 0.59%      | 3e-3          |
| Belavkin  | 17.65%        | 12.35% ± 5.29%      | 1e-3          |

### Analysis

**Key Observations**:
- No optimizer reached 90% accuracy with only 50 epochs
- Task exhibits "grokking" phenomenon - requires extended training
- Adam showed best performance but high variance
- Belavkin had highest variance across seeds

**Limitations**:
- Too few epochs for meaningful comparison
- Small modulus (p=13) makes task easier but less interesting
- Need to scale to p=97 with 200+ epochs

---

## Experiment 3: Modular Arithmetic - Full Scale (p=97)

### Setup
- **Task**: f(x,y) = (x + y) mod 97
- **Dataset**: 9409 examples (4704 train, 4705 test)
- **Model**: 2-layer MLP with [128, 128] hidden units
- **Training**: 150 epochs, batch size 512
- **Optimizers**: SGD, Adam, Belavkin, Adaptive Belavkin
- **Hyperparameters**:
  - Learning rates: [1e-3, 3e-3]
  - Gamma (Belavkin): 1e-4
  - Beta (Belavkin): 1e-2
- **Seeds**: 3

### Status
**RUNNING** - Results pending

**Expected Insights**:
- Will modular arithmetic's phase transitions reveal advantages for Belavkin?
- How does adaptive variant compare to fixed hyperparameters?
- Can we observe grokking behavior?

---

## Experiment 4: Ablation Study (p=97)

### Setup
- Testing contribution of each Belavkin component:
  1. **Full model**: All components enabled
  2. **SGD baseline**: Standard SGD for comparison
  3. **No damping**: γ = 0
  4. **No exploration**: β = 0
  5. **Additive noise**: β*ε instead of β*∇L*ε
  6. **No adaptation**: Fixed hyperparameters
  7. **Only damping**: Isolate damping effect
  8. **Only exploration**: Isolate stochastic effect

### Status
**RUNNING** - Results pending

**Research Questions**:
- Which component contributes most to performance?
- Is multiplicative noise better than additive?
- Do adaptive mechanisms help?

---

## Track 2: Reinforcement Learning

### Attempted Experiment: Noisy Gridworld

**Status**: **FAILED** - Missing dependency (gymnasium)

**Planned Setup**:
- 5×5 grid with goal-seeking task
- Observation noise: σ = 0.3
- Agent: Model-free Belavkin Q-learning
- Episodes: 500

**Next Steps**:
- Install gymnasium package
- Rerun experiment
- Compare model-based vs model-free variants

---

## Generated Artifacts

### Results Files
```
experiments/track1_optimizer/
├── benchmarks/
│   ├── modular_addition_p13/
│   │   └── benchmark_results.json
│   ├── modular_addition_p97/
│   │   └── [PENDING]
│   └── sparse_parity_n10_k3/
│       ├── benchmark_results.json
│       ├── sparse_parity_learning_curves.png
│       ├── sparse_parity_loss_curves.png
│       └── sparse_parity_comparison.png
└── ablations/
    └── modular_p97/
        └── [RUNNING]
```

### Visualizations Generated
- ✅ Sparse parity learning curves
- ✅ Sparse parity loss curves
- ✅ Optimizer comparison bar charts
- ⏳ Modular arithmetic p=97 (pending)
- ⏳ Ablation analysis (pending)

---

## Preliminary Conclusions

### What Works
1. **Implementation is functional**: All code runs without crashes
2. **Benchmarking infrastructure**: Automated comparison works well
3. **Visualization pipeline**: Automatic figure generation successful

### Performance Insights
1. **Adam dominates on sparse parity**: Belavkin offers no advantage for sparse structure discovery
2. **Modular arithmetic requires more investigation**: Preliminary results inconclusive due to insufficient training
3. **High variance observed**: Belavkin shows more variability across seeds than Adam

### Hypotheses for Further Testing
1. **Task dependency**: Belavkin may excel on continuous, smooth loss landscapes rather than discrete structure
2. **Hyperparameter sensitivity**: Current γ and β may not be optimal
3. **Scale effects**: Performance may improve/degrade with model size

---

## Next Steps

### Immediate (Hours)
1. ✅ Complete p=97 modular arithmetic benchmark
2. ✅ Complete ablation study
3. ✅ Analyze all results comprehensively

### Short-term (Days)
1. 📊 Generate paper-ready figures
2. 📝 Fill in results sections of LaTeX paper
3. 🔬 Run extended hyperparameter search if needed
4. 🎮 Fix and run Track 2 RL experiments

### Medium-term (Weeks)
1. 📈 Scale to larger tasks (MNIST, CIFAR-10)
2. 🧪 Test on continuous optimization benchmarks
3. 📄 Complete Track 1 manuscript
4. 🤖 Full Track 2 RL evaluation

---

## Research Implications

### For Paper Writing
- **Honest assessment**: Belavkin doesn't universally outperform Adam
- **Niche applications**: May need to identify specific scenarios where it helps
- **Theoretical contribution**: Even negative results contribute to understanding
- **Implementation value**: Open-source code enables further research

### For Future Work
- Consider continuous control tasks (may suit Belavkin better)
- Investigate connection to natural gradient methods
- Explore different parameterizations of damping and noise
- Test on non-convex optimization landscapes

---

## Computational Resources Used

- **CPU-only**: All experiments run on CPU (PyTorch CPU version)
- **Time per experiment**:
  - Sparse parity: ~2 minutes
  - Modular arithmetic p=13: ~40 seconds
  - Modular arithmetic p=97: ~5-10 minutes (estimated)
  - Ablation study: ~15-20 minutes (estimated)

---

## Reproducibility

All experiments used fixed seeds and are fully reproducible:
- Dataset seed: 42
- Training seeds: 0, 1, 2 (varied per run)
- All hyperparameters logged in result JSON files
- Code version: Commit d9d8df3

---

**Report Generated**: November 10, 2025
**Experiments**: Ongoing
**Status**: Preliminary results available, full results pending
