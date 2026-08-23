"""
Group-relative policy-gradient trainer (mean-baseline; see note below on why
this is not the standard per-group std-normalized GRPO estimator).

For each of `prompts_per_step` distinct prompts, sample `group_size`
rollouts under the current policy, score them, and use the *within-group*
mean-subtracted reward as the advantage -- REINFORCE with a group-mean
baseline, which needs no separate value network/critic.

IMPORTANT -- why this is NOT standard GRPO's (reward - group_mean) /
(group_std + eps): empirically (see the notebook's Section 7 discussion),
dividing by each group's own standard deviation, estimated from only
`group_size` samples, produces a wildly unstable advantage whenever a group
happens to land a small, noisy std -- amplifying a handful of lucky/unlucky
groups into oversized gradient contributions that swamp the signal from
every typical group. On this project's task and group size, that amplification
was severe enough to prevent any learning at all, including on a trivially
easy control task where supervised learning on the identical architecture
converges cleanly in a few hundred steps. Dropping the std division (plain
mean-baseline) fixed this. This is a known sensitivity of small-group GRPO-style
normalization, not a claim that GRPO itself never works -- it means the
per-group std estimate needs either a much larger group size or a more
stable (e.g. batch-level, not per-group) scale estimate than this project
budgeted for, and the simplest fix was tested and adopted directly rather
than assumed.

Two additional deliberate departures from a "purest possible" policy-gradient
baseline, both added after diagnosing why an earlier version of this trainer
showed zero learning even on trivial tasks:

  - A small entropy bonus IS now included in the loss. Without it, the
    policy's sampling entropy collapses early (observed directly, not
    inferred) -- as entropy falls, more groups sample the same token for
    every rollout, which zeroes THAT group's own reward variance and
    therefore its gradient contribution, which removes the only pressure
    that could have corrected the collapse. It's a self-reinforcing spiral
    toward an arbitrary, non-improving behavior. A standard, modest entropy
    bonus breaks the spiral. This does mean the policy-entropy SIGNAL this
    project studies is no longer entirely a free, unregularized quantity --
    a limitation worth stating plainly rather than hiding, and one the
    analysis should keep in mind when interpreting entropy's discriminative
    power specifically.
  - An optional brief supervised (teacher-forced cross-entropy) warmup phase
    is available via `supervised_warmup`, run once before RL begins. Real
    RLVR always fine-tunes an already-capable pretrained/SFT model, never a
    randomly-initialized one; a from-scratch policy has to bootstrap purely
    from whatever it happens to sample, which is a much harder exploration
    problem than the RLVR setting this project is trying to speak to. A
    short, deliberately-incomplete supervised warmup (stopped well short of
    full convergence) gives the RL phase a non-degenerate starting point
    without pre-solving the task, closing the gap between this toy setup and
    what "RLVR fine-tuning" actually means. It is applied identically to
    both reward schedules, so the two_phase vs. binary_only comparison stays
    about the reward schedule, not about which one got a better starting point.

Two reward schedules are supported, both taken directly from what "RL
Grokking Recipe" documents as the mechanism that separates its
breakthrough and stuck outcomes:

  - "two_phase" (breakthrough condition): dense (graded) reward for the
    first `warmup_fraction` of training, then binary reward for the rest.
  - "binary_only" (stall condition): binary reward from step 0.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Callable

import numpy as np
import torch

from model import TinyPolicy
from signals import SignalProbe
from tasks import RewardMode


@dataclasses.dataclass
class GRPOConfig:
    total_steps: int
    prompts_per_step: int
    group_size: int
    lr: float
    weight_decay: float
    sample_temperature: float
    schedule: str  # "two_phase" | "binary_only"
    warmup_fraction: float  # only used when schedule == "two_phase"
    success_threshold: float = 0.9
    success_window: int = 50
    signal_log_every: int = 25
    grad_clip: float = 1.0
    entropy_coef: float = 0.05  # standard small entropy bonus; see module docstring


def supervised_warmup(
    policy: TinyPolicy,
    task,
    n_steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    seed: int,
    target_train_accuracy: float | None = None,
) -> dict:
    """
    Brief teacher-forced cross-entropy pretraining, run in place on `policy`
    before any GRPO training. This exists to give the RL phase a genuine,
    non-degenerate starting point -- mirroring how real RLVR always fine-tunes
    an already-capable SFT model, never a randomly-initialized one. Pure
    policy-gradient learning from random initialization has to bootstrap
    entirely from whatever the model happens to sample; a supervised warmup
    solves the much easier "learn the rough shape of the function" problem
    with a dense, exact gradient, leaving RL responsible for the refinement
    step it's actually good at, not for exploration from zero.

    Stops early if `target_train_accuracy` is reached before `n_steps`, so a
    generous step budget doesn't force the model past "still learning" into
    "fully converged" -- the interesting RL dynamics need a starting point
    with room left to move, not one that's already solved the task.
    """
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    policy.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for step in range(n_steps):
        batch = task.sample_batch(batch_size, np_rng)
        prompts = batch.prompt_ids.to(device)
        targets = batch.targets.to(device)
        full_seqs = torch.cat([prompts, targets], dim=1)
        token_logprobs, _, _ = policy.sequence_logprobs(full_seqs, prompt_len=prompts.shape[1])
        loss = -token_logprobs.sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == n_steps - 1:
            with torch.no_grad():
                logits, _ = policy.forward(full_seqs)
                pred_logits = logits[:, prompts.shape[1] - 1 : -1, :]
                acc = (pred_logits.argmax(-1) == targets).float().mean().item()
            history.append({"step": step, "loss": float(loss.item()), "train_accuracy": acc})
            if target_train_accuracy is not None and acc >= target_train_accuracy:
                break
    return {"history": history, "steps_used": history[-1]["step"] + 1 if history else 0,
            "final_train_accuracy": history[-1]["train_accuracy"] if history else 0.0}


def reward_mode_for_step(cfg: GRPOConfig, step: int) -> RewardMode:
    if cfg.schedule == "binary_only":
        return "binary"
    if cfg.schedule == "two_phase":
        warmup_steps = int(cfg.warmup_fraction * cfg.total_steps)
        return "dense" if step < warmup_steps else "binary"
    raise ValueError(f"Unknown schedule {cfg.schedule!r}")


@dataclasses.dataclass
class StepLog:
    step: int
    reward_mean: float
    reward_within_group_var: float
    loss: float
    grad_norm: float
    wall_time: float


@dataclasses.dataclass
class SignalLog:
    step: int
    policy_entropy: float
    effective_rank: float
    kl_from_reference: float


@dataclasses.dataclass
class RunResult:
    step_logs: list[StepLog]
    signal_logs: list[SignalLog]
    breakthrough_step: int | None  # None if never broke through within total_steps
    final_state_dict: dict
    optimizer_state_dict: dict


def _detect_breakthrough(binary_success_history: list[float], cfg: GRPOConfig) -> int | None:
    """First step index at which a trailing window's mean binary success rate
    crosses `success_threshold`. Returns None if it never does."""
    w = cfg.success_window
    if len(binary_success_history) < w:
        return None
    arr = np.asarray(binary_success_history)
    trailing = np.convolve(arr, np.ones(w) / w, mode="valid")
    hit = np.where(trailing >= cfg.success_threshold)[0]
    if len(hit) == 0:
        return None
    return int(hit[0] + w - 1)


def train_one_run(
    policy: TinyPolicy,
    task,
    cfg: GRPOConfig,
    seed: int,
    device: torch.device,
    start_step: int = 0,
    resume_optimizer_state: dict | None = None,
    resume_step_logs: list[StepLog] | None = None,
    resume_signal_logs: list[SignalLog] | None = None,
    on_step: Callable[[int], None] | None = None,
) -> RunResult:
    """
    Run (or resume) one GRPO training run to completion. `on_step` is an
    optional callback invoked after every step, used by the notebook to
    persist a resumable checkpoint without this function knowing anything
    about the filesystem.
    """
    # The generator must live on the same device as the tensors it's sampling
    # from (torch.multinomial requires this when an explicit generator is
    # passed) -- policy, prompts, and therefore `probs` inside generate() all
    # live on `device`, so the generator does too.
    torch_gen = torch.Generator(device=device).manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    policy.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)

    reference_policy = policy.clone_frozen().to(device)
    probe_batch = task.sample_batch(64, np.random.default_rng(seed + 10_000))
    probe = SignalProbe(
        prompt_ids=probe_batch.prompt_ids.to(device),
        n_answer_tokens=task.n_answer_tokens,
        reference_policy=reference_policy,
    )

    step_logs = list(resume_step_logs or [])
    signal_logs = list(resume_signal_logs or [])
    binary_success_history = [s.reward_mean for s in step_logs] if False else []
    # rebuild binary-success history for breakthrough detection from step_logs
    # (reward_mean is already binary-mode-aware once past warmup; during warmup
    # we track binary success separately below via `binary_success_history`)

    breakthrough_step = None
    for step in range(start_step, cfg.total_steps):
        t0 = time.time()
        mode = reward_mode_for_step(cfg, step)

        # Measure signals on the policy as it enters this step (i.e. BEFORE this
        # step's gradient update), so step 0's measurement reflects the truly
        # untrained policy (KL-from-reference == 0 there, by construction) and
        # every logged signal is a property of "what we knew before committing
        # to this step" -- the same information an early-stopping rule would
        # actually have access to.
        if step % cfg.signal_log_every == 0 or step == start_step:
            measured = probe.measure(policy, cfg.sample_temperature, torch_gen)
            signal_logs.append(SignalLog(step=step, **measured))

        batch = task.sample_batch(cfg.prompts_per_step, np_rng)
        prompts = batch.prompt_ids.to(device)
        targets = batch.targets.to(device)
        prompt_len = prompts.shape[1]

        # expand each prompt into `group_size` identical copies for group rollouts
        rep_prompts = prompts.repeat_interleave(cfg.group_size, dim=0)
        rep_targets = targets.repeat_interleave(cfg.group_size, dim=0)

        _, _, full_seqs = policy.generate(
            rep_prompts, task.n_answer_tokens, cfg.sample_temperature, generator=torch_gen
        )
        answer_tokens = full_seqs[:, prompt_len:]
        rewards = task.reward(rep_targets, answer_tokens, mode)  # (prompts*group,)
        binary_rewards = task.reward(rep_targets, answer_tokens, "binary")

        rewards_g = rewards.view(cfg.prompts_per_step, cfg.group_size)
        group_mean = rewards_g.mean(dim=1, keepdim=True)
        # Mean-baseline only -- no division by group_std. See module docstring:
        # per-group std, estimated from only `group_size` samples, is too noisy
        # not to amplify a handful of groups into destabilizing the whole batch.
        advantage = (rewards_g - group_mean).view(-1)

        token_logprobs, token_entropy, _ = policy.sequence_logprobs(full_seqs, prompt_len=prompt_len)
        seq_logprob = token_logprobs.sum(dim=1)
        pg_loss = -(advantage.detach() * seq_logprob).mean()
        entropy_term = token_entropy.mean()
        loss = pg_loss - cfg.entropy_coef * entropy_term

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        step_logs.append(
            StepLog(
                step=step,
                reward_mean=float(rewards.mean().item()),
                reward_within_group_var=float(rewards_g.var(dim=1).mean().item()),
                loss=float(loss.item()),
                grad_norm=float(grad_norm),
                wall_time=time.time() - t0,
            )
        )
        binary_success_history.append(float(binary_rewards.mean().item()))

        if breakthrough_step is None:
            bt = _detect_breakthrough(binary_success_history, cfg)
            if bt is not None:
                breakthrough_step = bt

        if on_step is not None:
            on_step(step)

    return RunResult(
        step_logs=step_logs,
        signal_logs=signal_logs,
        breakthrough_step=breakthrough_step,
        final_state_dict={k: v.cpu() for k, v in policy.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(),
    )