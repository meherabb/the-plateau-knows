"""
The four candidate early-warning signals, plus the fixed evaluation probe
used to measure them off-policy (i.e. without disturbing training).

All four are computed on the SAME fixed probe batch at every logging
interval, sampled once at run start and held fixed for the life of the
run, so that trajectories are comparable across steps and are not
confounded by which prompts happened to be drawn that step.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from model import TinyPolicy


def effective_rank(hidden_states: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Effective rank of a set of representation vectors (Roy & Vetterli 2007):
    exponential of the Shannon entropy of the normalized singular value
    spectrum of the (n_vectors, dim) matrix. Bounded in [1, min(n, dim)].
    """
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).detach()
    x = x - x.mean(dim=0, keepdim=True)  # center, as is standard for this statistic
    # SVD on CPU float32 for numerical stability regardless of training dtype/device
    s = torch.linalg.svdvals(x.float().cpu())
    p = s / (s.sum() + eps)
    p = p.clamp_min(eps)
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


@dataclasses.dataclass
class SignalProbe:
    """Fixed evaluation batch + reference policy snapshot used to measure signals."""
    prompt_ids: torch.Tensor
    n_answer_tokens: int
    reference_policy: TinyPolicy

    @torch.no_grad()
    def measure(self, policy: TinyPolicy, temperature: float, rng: torch.Generator) -> dict:
        """
        One measurement of (policy entropy, effective rank) on the fixed probe,
        off-policy and without affecting training. KL-from-reference and
        reward variance are NOT computed here: KL needs matched positions
        between policy and reference (done here too, see below), reward
        variance is instead logged directly from each training step's
        on-policy rollouts (see grpo.py), since it is a property of the
        *training* batch, not a fixed probe.
        """
        policy.eval()
        gen_tokens, gen_entropy, full_seqs = policy.generate(
            self.prompt_ids, self.n_answer_tokens, temperature, generator=rng
        )
        _, _, hidden = policy.sequence_logprobs(full_seqs, prompt_len=self.prompt_ids.shape[1])

        # KL(policy || reference) at the same probe positions, teacher-forced
        # on the policy's own sampled continuation so both models are scored
        # on the same token positions.
        logits_policy, _ = policy.forward(full_seqs)
        logits_ref, _ = self.reference_policy.forward(full_seqs)
        p_len = self.prompt_ids.shape[1]
        log_p = F.log_softmax(logits_policy[:, p_len - 1 : -1, :], dim=-1)
        log_q = F.log_softmax(logits_ref[:, p_len - 1 : -1, :], dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean().item()

        policy.train()
        return {
            "policy_entropy": gen_entropy.mean().item(),
            "effective_rank": effective_rank(hidden),
            "kl_from_reference": kl,
        }