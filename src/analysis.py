"""
Statistical analysis of run outcomes: does an early-window signal value
discriminate breakthrough runs from stalled runs, with how much lead time,
and does it carry information beyond simply knowing the run's current raw
reward (the partial-correlation functions, added specifically to test
whether a signal is "just reward in disguise")?

Kept dependency-light and self-contained (numpy + scipy + sklearn only) so
it is easy to audit and does not depend on a heavyweight mixed-effects
modeling library that may not be preinstalled on a fresh Kaggle image.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


@dataclasses.dataclass
class RunSummary:
    """One completed training run, reduced to what the analysis needs."""
    run_id: str
    task: str
    schedule: str  # "two_phase" | "binary_only"
    seed: int
    breakthrough_step: int | None
    total_steps: int
    signal_trajectory: dict  # signal_name -> list[(step, value)]
    final_reward: float | None = None  # mean reward over the last few logged
                                        # steps -- a continuous outcome usable
                                        # for EVERY run, not just the subset
                                        # that crossed the binary breakthrough
                                        # threshold (see continuous_outcome_correlation)

    @property
    def broke_through(self) -> bool:
        return self.breakthrough_step is not None


def early_window_value(trajectory: list[tuple[int, float]], window_fraction: float, total_steps: int) -> float | None:
    """Mean of the signal over the first `window_fraction` of total_steps. None if no points fall in the window."""
    cutoff = window_fraction * total_steps
    vals = [v for (s, v) in trajectory if s <= cutoff]
    if not vals:
        return None
    return float(np.mean(vals))


def residualize(xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """OLS residuals of xs regressed on zs (plus intercept): the part of xs
    NOT linearly explained by zs. Used to test whether a candidate signal
    carries information beyond a baseline predictor (e.g. current raw
    reward) rather than just correlating with it."""
    slope, intercept = np.polyfit(zs, xs, deg=1)
    return xs - (slope * zs + intercept)


def _auroc_with_stats(xs: np.ndarray, ys: np.ndarray, n_bootstrap: int, seed: int, extra: dict) -> dict:
    """Shared AUROC + bootstrap-CI + permutation-p-value core, used by both
    discrimination_auroc and partial_discrimination_auroc so the two stay
    numerically consistent with each other."""
    if len(set(ys.tolist())) < 2 or len(xs) < 4:
        return {**extra, "n_runs": len(xs), "auroc": float("nan"),
                "auroc_ci_low": float("nan"), "auroc_ci_high": float("nan"), "p_value": float("nan"),
                "note": "insufficient class balance or sample size for AUROC"}

    auroc = roc_auc_score(ys, xs)

    rng = np.random.default_rng(seed)
    boot = []
    n = len(xs)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb, xb = ys[idx], xs[idx]
        if len(set(yb.tolist())) < 2:
            continue
        boot.append(roc_auc_score(yb, xb))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))

    perm_stats = []
    for _ in range(n_bootstrap):
        yp = rng.permutation(ys)
        if len(set(yp.tolist())) < 2:
            continue
        perm_stats.append(roc_auc_score(yp, xs))
    perm_stats = np.asarray(perm_stats)
    observed_dev = abs(auroc - 0.5)
    p_value = float(np.mean(np.abs(perm_stats - 0.5) >= observed_dev)) if len(perm_stats) else float("nan")

    return {**extra, "n_runs": len(xs), "n_breakthrough": int(ys.sum()), "n_stall": int((1 - ys).sum()),
            "auroc": float(auroc), "auroc_ci_low": float(ci_low), "auroc_ci_high": float(ci_high), "p_value": p_value}


def _spearman_with_stats(xs: np.ndarray, ys: np.ndarray, n_bootstrap: int, seed: int, extra: dict) -> dict:
    """Shared Spearman-r + bootstrap-CI + permutation-p-value core, used by
    both continuous_outcome_correlation and partial_continuous_outcome_correlation."""
    if len(xs) < 4:
        return {**extra, "n_runs": len(xs), "spearman_r": float("nan"),
                "spearman_ci_low": float("nan"), "spearman_ci_high": float("nan"), "p_value": float("nan"),
                "note": "insufficient sample size for correlation"}

    r_obs, _ = spearmanr(xs, ys)

    rng = np.random.default_rng(seed)
    boot = []
    n = len(xs)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(set(xs[idx].tolist())) < 2 or len(set(ys[idx].tolist())) < 2:
            continue
        rb, _ = spearmanr(xs[idx], ys[idx])
        if np.isfinite(rb):
            boot.append(rb)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))

    perm_stats = []
    for _ in range(n_bootstrap):
        yp = rng.permutation(ys)
        rp, _ = spearmanr(xs, yp)
        if np.isfinite(rp):
            perm_stats.append(rp)
    perm_stats = np.asarray(perm_stats)
    p_value = float(np.mean(np.abs(perm_stats) >= abs(r_obs))) if len(perm_stats) else float("nan")

    return {**extra, "n_runs": len(xs), "spearman_r": float(r_obs),
            "spearman_ci_low": float(ci_low), "spearman_ci_high": float(ci_high), "p_value": p_value}


def discrimination_auroc(
    runs: list[RunSummary], signal_name: str, window_fraction: float, n_bootstrap: int = 2000, seed: int = 0
) -> dict:
    """
    AUROC of "signal's mean value during the first `window_fraction` of
    training" for predicting eventual breakthrough (1) vs stall (0), plus a
    bootstrap CI and a permutation-test p-value against the null of no
    discrimination.
    """
    xs, ys = [], []
    for r in runs:
        v = early_window_value(r.signal_trajectory[signal_name], window_fraction, r.total_steps)
        if v is None or not np.isfinite(v):
            continue
        xs.append(v)
        ys.append(1 if r.broke_through else 0)
    extra = {"signal": signal_name, "window_fraction": window_fraction}
    return _auroc_with_stats(np.asarray(xs), np.asarray(ys), n_bootstrap, seed, extra)


def continuous_outcome_correlation(
    runs: list[RunSummary], signal_name: str, window_fraction: float, n_bootstrap: int = 2000, seed: int = 0
) -> dict:
    """
    Spearman correlation between an early-window signal value and each run's
    FINAL reward -- continuous, and computed using EVERY run regardless of
    whether it crossed the binary breakthrough threshold. This is the
    higher-power complement to discrimination_auroc: the AUROC analysis only
    uses a run's breakthrough/stall label, throwing away graded information
    for every run that improved substantially without quite crossing the
    (somewhat arbitrary) sustained-90%-success threshold. Bootstrap CI and a
    permutation-test p-value included, same conventions as discrimination_auroc.
    """
    xs, ys = [], []
    for r in runs:
        v = early_window_value(r.signal_trajectory[signal_name], window_fraction, r.total_steps)
        if v is None or not np.isfinite(v) or r.final_reward is None or not np.isfinite(r.final_reward):
            continue
        xs.append(v)
        ys.append(r.final_reward)
    extra = {"signal": signal_name, "window_fraction": window_fraction}
    return _spearman_with_stats(np.asarray(xs), np.asarray(ys), n_bootstrap, seed, extra)


def partial_discrimination_auroc(
    runs: list[RunSummary], signal_name: str, baseline_values: dict, window_fraction: float,
    n_bootstrap: int = 2000, seed: int = 0
) -> dict:
    """
    Same as discrimination_auroc, but on the signal's value AFTER
    residualizing out a baseline predictor (baseline_values: run_id -> float,
    e.g. raw reward at the same early window) -- answers "does this signal
    predict breakthrough beyond what the baseline already tells you", not
    just "is this signal correlated with breakthrough" (which the baseline
    alone might already explain). This is the direct test of the "is this
    just reward in disguise" objection.
    """
    xs, zs, ys = [], [], []
    for r in runs:
        v = early_window_value(r.signal_trajectory[signal_name], window_fraction, r.total_steps)
        b = baseline_values.get(r.run_id)
        if v is None or b is None or not np.isfinite(v) or not np.isfinite(b):
            continue
        xs.append(v)
        zs.append(b)
        ys.append(1 if r.broke_through else 0)
    xs, zs, ys = np.asarray(xs), np.asarray(zs), np.asarray(ys)
    extra = {"signal": signal_name, "window_fraction": window_fraction}
    if len(xs) < 6 or len(set(zs.tolist())) < 2:
        return {**extra, "n_runs": len(xs), "auroc": float("nan"), "auroc_ci_low": float("nan"),
                "auroc_ci_high": float("nan"), "p_value": float("nan"),
                "note": "insufficient sample size or baseline variance to residualize"}
    residuals = residualize(xs, zs)
    return _auroc_with_stats(residuals, ys, n_bootstrap, seed, extra)


def partial_continuous_outcome_correlation(
    runs: list[RunSummary], signal_name: str, baseline_values: dict, window_fraction: float,
    n_bootstrap: int = 2000, seed: int = 0
) -> dict:
    """Continuous-outcome counterpart to partial_discrimination_auroc: Spearman
    correlation between the signal's baseline-residualized value and final
    reward, using every run."""
    xs, zs, ys = [], [], []
    for r in runs:
        v = early_window_value(r.signal_trajectory[signal_name], window_fraction, r.total_steps)
        b = baseline_values.get(r.run_id)
        if (v is None or b is None or not np.isfinite(v) or not np.isfinite(b)
                or r.final_reward is None or not np.isfinite(r.final_reward)):
            continue
        xs.append(v)
        zs.append(b)
        ys.append(r.final_reward)
    xs, zs, ys = np.asarray(xs), np.asarray(zs), np.asarray(ys)
    extra = {"signal": signal_name, "window_fraction": window_fraction}
    if len(xs) < 6 or len(set(zs.tolist())) < 2:
        return {**extra, "n_runs": len(xs), "spearman_r": float("nan"), "spearman_ci_low": float("nan"),
                "spearman_ci_high": float("nan"), "p_value": float("nan"),
                "note": "insufficient sample size or baseline variance to residualize"}
    residuals = residualize(xs, zs)
    return _spearman_with_stats(residuals, ys, n_bootstrap, seed, extra)


def lead_time_steps(run: RunSummary, signal_name: str, threshold_z: float = 2.0) -> int | None:
    """
    Steps between the first point the signal crosses `threshold_z` standard
    deviations away from its own first-quartile-of-training baseline, and
    the run's breakthrough step. None if the run never broke through or the
    signal never crosses the threshold before breakthrough.
    """
    if not run.broke_through:
        return None
    traj = sorted(run.signal_trajectory[signal_name])
    steps = np.array([s for s, _ in traj])
    vals = np.array([v for _, v in traj])
    baseline_mask = steps <= 0.25 * run.total_steps
    if baseline_mask.sum() < 2:
        return None
    mu, sigma = vals[baseline_mask].mean(), vals[baseline_mask].std() + 1e-8
    z = np.abs((vals - mu) / sigma)
    before_breakthrough = steps < run.breakthrough_step
    crossing = np.where(before_breakthrough & (z >= threshold_z))[0]
    if len(crossing) == 0:
        return None
    first_cross_step = steps[crossing[0]]
    return int(run.breakthrough_step - first_cross_step)
