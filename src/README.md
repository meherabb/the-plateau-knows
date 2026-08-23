# `src/`

Standalone Python source, extracted verbatim from the cells the notebook
writes to disk at runtime (`%%writefile src/*.py`) — this is exactly the code
that produced every result in the paper, not a re-implementation.

| Module | Contents |
|---|---|
| `tasks.py` | The two fixed-pair modular-arithmetic task families (Task A: single-token, Task B: two-token), the train/test split logic, and the task-integrity self-check that verifies empirical random-policy success rate against the theoretical value before any training begins. |
| `model.py` | `TinyPolicy` — the causal decoder-only transformer (821,504 parameters) used as the RL policy throughout. |
| `grpo.py` | The GRPO trainer: the group-mean-baseline advantage estimator (§3.2.1 of the paper — the fix that replaces standard GRPO's per-group std-normalized advantage, which we found prevents any learning at this group size), the entropy bonus that prevents a sampling-collapse spiral, the two reward schedules (`two_phase` / `binary_only`), and the supervised warmup routine. |
| `signals.py` | The four candidate signal probes — `policy_entropy`, `effective_rank` (Roy & Vetterli, 2007), `kl_from_reference`, `reward_variance` — each measured on a fixed evaluation probe sampled once at run start. |
| `analysis.py` | The complete statistical methodology: `discrimination_auroc`, `continuous_outcome_correlation`, `partial_discrimination_auroc`, `partial_continuous_outcome_correlation`, `residualize`, and `lead_time_steps`, each with bootstrap confidence intervals and permutation-test p-values. |

These modules are dependency-light (`numpy`, `scipy`, `scikit-learn`, `torch`
only) and are usable independently of the notebook — for example, to rerun
`analysis.py`'s tests against the logs in `results/raw_run_logs/` without
retraining anything.
