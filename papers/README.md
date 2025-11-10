# Paper Writing Infrastructure

This directory contains LaTeX templates and resources for writing papers based on the Belavkin ML research.

## Structure

```
papers/
├── optimizer/          # Track 1: Belavkin Optimizer paper
│   ├── main.tex       # Main paper file
│   ├── abstract.tex   # Abstract
│   ├── intro.tex      # Introduction
│   ├── method.tex     # Methodology
│   ├── experiments.tex # Experiments
│   ├── results.tex    # Results
│   ├── discussion.tex # Discussion
│   └── references.bib # Bibliography
├── rl/                # Track 2: Belavkin RL paper
│   └── (same structure)
├── theory/            # Optional: Theoretical analysis paper
│   └── (same structure)
└── shared/            # Shared resources
    ├── figures/       # Shared figures
    └── macros.tex     # Shared macros
```

## Target Venues

- **NeurIPS** (Neural Information Processing Systems)
- **ICML** (International Conference on Machine Learning)
- **ICLR** (International Conference on Learning Representations)
- **JMLR** (Journal of Machine Learning Research) for longer versions

## Compilation

```bash
# Compile Track 1 paper
cd papers/optimizer
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or use latexmk for automatic compilation
latexmk -pdf main.tex
```

## Writing Guidelines

1. **Be precise**: Use mathematical notation consistently
2. **Be honest**: Report negative results and limitations
3. **Be reproducible**: Include all hyperparameters and seeds
4. **Be clear**: Avoid quantum jargon; explain intuitions
5. **Be modest**: Avoid overstating claims about quantum connections

## Paper Checklist

Before submission:

- [ ] All experiments completed and results included
- [ ] Ablation studies conducted
- [ ] Code released (or will be upon acceptance)
- [ ] Ethical considerations addressed
- [ ] Reproducibility checklist completed
- [ ] All figures have captions and are referenced in text
- [ ] All tables are formatted properly
- [ ] References are complete and properly formatted
- [ ] Appendix includes full experimental details
- [ ] Proofread for typos and clarity
