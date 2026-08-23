<div align="center">

# The Plateau Knows

### Pre-Registered Evidence for Early Predictors of RL Training Outcomes

*Anonymous authors — submitted for double-blind review*

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/main.tex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/pipeline-fully%20resumable-brightgreen.svg)](notebooks/the_plateau_knows.ipynb)

</div>

---

## Overview

Grokking — a long plateau at chance performance followed by an abrupt jump to high
accuracy — is well studied in supervised learning, where several signals are known to
precede the transition. A recent line of work shows an analogous plateau-then-breakthrough
pattern occurs during **reinforcement learning with verifiable rewards (RLVR)**, driven by
reward schedule rather than epoch count. This repository contains the complete,
reproducible pipeline behind a pre-registered, three-round study asking a question that,
to our knowledge, had not been directly tested: **do the early-warning signals identified
for supervised-learning grokking transfer to this qualitatively different RL setting?**

Along the way, we identified and fixed a training-stability failure mode in group relative
policy optimization (GRPO) that was silently preventing any breakthroughs from occurring
at our scale — a precondition for the scientific question to be askable at all.

**We report our findings honestly, including the ones that didn't confirm our hypotheses.**
A pre-registered signal from an early pilot (`reward_variance`) failed to replicate under a
properly powered follow-up, and we say so directly rather than substituting a
better-looking result. A second pre-registered signal (`effective_rank`) predicts a run's
eventual training reward robustly — independently replicated across two disjoint seed
batches, and surviving a check for whether it's just reflecting current reward level — but
its ability to predict the *discrete* breakthrough event that originally motivated the
question does **not** survive the same held-out replication test. We take that result at
face value rather than explain it away, and it is the paper's central finding.

## Key Results

| Test (pre-registered primary: `effective_rank` @ window 0.1) | Statistic | 95% CI | *p* |
|---|---|---|---|
| Discrimination (breakthrough vs. stall) | AUROC = 0.730 | [0.508, 0.890] | 0.0385 |
| Continuous outcome (final reward) | *r* = 0.647 | [0.489, 0.756] | < 0.001 |
| Discrimination, controlling for raw reward | AUROC = 0.695 | [0.506, 0.852] | 0.0870 |
| Continuous outcome, controlling for raw reward | *r* = 0.630 | [0.478, 0.751] | < 0.001 |

**The held-out replication check that drives the paper's honest framing:**

| Seed batch | *n* (breakthroughs) | AUROC (*p*) | *r* (*p*) |
|---|---|---|---|
| 0–23 (the batch that originally suggested this signal) | 48 (6) | 0.790 (0.018) | 0.615 (< 0.001) |
| 24–47 (the true, pre-registered confirmatory batch) | 48 (1) | **0.277 (0.574)** | **0.663 (< 0.001)** |
| Both combined | 96 (7) | 0.730 (0.039) | 0.647 (< 0.001) |

On the genuinely out-of-sample seed batch, discrete-breakthrough discrimination is *below
chance* and far from significant — the pooled significance is substantially carried by the
non-independent selection batch. The continuous-outcome correlation, in contrast,
replicates cleanly and independently in both batches. See [`paper/main.tex`](paper/main.tex),
§5.3, for the full discussion of why we treat this as the paper's central finding rather
than a nuisance to write around.

<p align="center">
  <img src="results/figures/fig7_primary_result_scatter.pdf" width="85%" alt="Primary result: effective_rank vs. final reward, raw and reward-controlled">
</p>

<p align="center"><sub><b>Figure.</b> <code>effective_rank</code> at the early plateau against eventual final
reward, raw (left) and after residualizing out current reward (right), colored by eventual
breakthrough outcome. The relationship survives residualization essentially unchanged.</sub></p>

## What's in This Repository

```
.
├── paper/                  Full LaTeX source, bibliography, and figures for the submission
│   ├── main.tex
│   ├── references.bib
│   ├── iclr2027_conference.sty
│   └── figures/
├── src/                    Standalone, importable Python source for the entire pipeline
│   ├── tasks.py            Fixed-pair modular-arithmetic task definitions (Task A / Task B)
│   ├── model.py             TinyPolicy — the causal decoder-only transformer policy
│   ├── grpo.py              GRPO trainer, including the mean-baseline advantage fix (§3.2.1)
│   ├── signals.py           The four candidate signal probes (entropy, effective rank, KL, reward variance)
│   └── analysis.py          Full statistical methodology: AUROC, Spearman, bootstrap CI,
│                            permutation tests, partial correlation / residualization
├── notebooks/
│   └── the_plateau_knows.ipynb   The complete, executed, end-to-end pipeline (74 cells, 0 errors) —
│                                  configuration, calibration, training sweep, statistical analysis,
│                                  and every figure/table in the paper, generated in order
├── results/
│   ├── figures/             All 8 paper figures (vector PDF, publication-ready)
│   ├── tables/               All 16 result tables (CSV + camera-ready LaTeX)
│   ├── metrics/               Raw statistical outputs: every AUROC, correlation, CI, and
│   │                           p-value reported in the paper, including secondary/exploratory
│   │                           tests that did not confirm the hypotheses under test
│   ├── configs/                Exact configuration used to produce the reported results
│   ├── environment/            Full dependency/environment fingerprint (Python, CUDA, package versions)
│   ├── summaries/               Pipeline self-validation checks (15/15 passed)
│   └── raw_run_logs/            Per-step signal and reward trajectories for all 144 independent
│                                 training runs (compressed archive — see below)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Reproducing the Results

The entire pipeline — task definitions, model, training, statistical analysis, and every
figure and table in the paper — is a single, linearly-executable, fully resumable Jupyter
notebook: [`notebooks/the_plateau_knows.ipynb`](notebooks/the_plateau_knows.ipynb).

```bash
git clone <this-repository>
cd the-plateau-knows
pip install -r requirements.txt
jupyter notebook notebooks/the_plateau_knows.ipynb
```

Designed for a **single consumer/free-tier GPU** (developed and run end-to-end on a single
NVIDIA T4, ~4.4 hours wall-clock for the full 144-run pipeline). The notebook:

- Auto-calibrates its own step budget against measured per-step throughput and a
  configurable wall-clock ceiling, so it adapts to whatever GPU it's run on.
- Is checkpoint-level resumable: interrupting and re-running only retrains what's
  missing, validated against a training-relevant configuration hash so that extending
  the seed count (as done between Round 2 and Round 3 of the pre-registration protocol)
  does not invalidate or retrain already-completed runs.
- Reproduces every number in the paper from logged, machine-generated output — nothing
  in `paper/main.tex` is manually transcribed, estimated, or rounded beyond the precision
  shown.

The `src/` modules mirror exactly what the notebook writes to disk at runtime and are
provided standalone for readers who want to import the trainer, the signal probes, or the
statistical analysis functions (`discrimination_auroc`, `continuous_outcome_correlation`,
`partial_discrimination_auroc`, `residualize`, ...) independently of the notebook.

### Raw per-run logs

`results/raw_run_logs/all_144_runs_logs.tar.gz` contains the full per-step signal and
reward trajectory for every one of the 144 independent training runs behind this paper
(both task families, both reward schedules, all seeds) — everything needed to recompute
any statistic in the paper from scratch without retraining:

```bash
tar -xzf results/raw_run_logs/all_144_runs_logs.tar.gz -C results/raw_run_logs/
```

Trained model weights are not included in this repository (144 runs × ~3.3MB), but are
fully reproducible by re-running the notebook, which is resumable and will not
retrain any run whose logs are already present.

## Method Summary

- **Tasks.** Two fixed-pair modular-arithmetic task families ($97 \times 97$ pairs,
  50/50 train/test split), deliberately matching the classic supervised-grokking setup —
  chosen after diagnosing that an earlier, larger input space could not support the
  memorize-then-generalize mechanism grokking depends on (§3.1).
- **The GRPO fix.** Standard GRPO's per-group standard-deviation-normalized advantage
  prevents *any* learning at our group size and reward regime — verified directly against
  supervised learning on an identical architecture and task. Removing the normalization
  term (a plain group-mean baseline, in the spirit of REINFORCE) restores learning (§3.2.1).
- **Four candidate signals**, each measured only during the early, near-chance plateau:
  policy entropy, effective rank of hidden-state representations (Roy & Vetterli, 2007),
  KL-divergence from a reference policy, and within-group reward variance (§3.3).
- **Statistical methodology.** Discrimination AUROC with bootstrap confidence intervals
  and permutation-test p-values; a continuous-outcome Spearman correlation against final
  reward that uses every run, not only the ones that crossed the breakthrough threshold;
  partial correlation via OLS residualization to test whether a signal predicts outcomes
  beyond current reward level; Bonferroni-corrected secondary/exploratory reporting (§3.4).
- **Three-round pre-registration protocol**, with the full history — including a
  reported failure to replicate — released as part of this repository, not just the
  final round (§3.4.1).

Full details, all four candidate signals' complete results (including Bonferroni-corrected
secondary tables), the seed-clustering finding, the cross-task-family null result, and
every stated limitation are in [`paper/main.tex`](paper/main.tex).

## Citation

This work is under double-blind review. Please cite as:

```bibtex
@inproceedings{anonymous2027plateau,
  title     = {The Plateau Knows: Pre-Registered Evidence for Early Predictors of {RL} Training Outcomes},
  author    = {Anonymous},
  booktitle = {Under double-blind review},
  year      = {2027}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## License

Code released under the [MIT License](LICENSE). See individual files for any third-party
attributions.

## Reproducibility and Integrity Statement

Every number in the paper traces to a specific logged cell output, checkpoint file, or
exported CSV/JSON artifact in this repository — none are estimated, interpolated, or
rounded beyond the precision shown. This includes every statistical test the pipeline ran,
not only the ones that confirmed the hypotheses under test: the Round 2 pre-registered
signal's failure to replicate, the held-out-batch result that narrows the paper's central
claim, the cross-task-family null result, and an attempted (incomplete) cross-architecture
check are all reported and included exactly as produced.
