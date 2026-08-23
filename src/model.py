"""
A tiny decoder-only transformer used as the RL policy.

Deliberately minimal: this is the same scale of model used in the
supervised-grokking notebooks this project builds on (single-digit-million
parameters, a handful of layers), just trained with policy gradients
instead of teacher-forced cross-entropy. Answer generation is short
(1 or 2 tokens depending on the task -- see tasks.py), so a full KV-cache
is unnecessary; each generation step simply re-runs the (very short)
forward pass, which is simpler to get correct and fast enough at this
scale.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 16
    dropout: float = 0.0  # grokking-style setups typically train without dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, n_heads, t, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TinyPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """token_ids: (batch, seq_len) -> logits (batch, seq_len, vocab), hidden (batch, seq_len, d_model)."""
        b, t = token_ids.shape
        pos = torch.arange(t, device=token_ids.device).unsqueeze(0)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        hidden = self.ln_f(x)
        logits = self.head(hidden)
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        n_new_tokens: int,
        temperature: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample `n_new_tokens` continuation tokens per row of prompt_ids.
        Returns (sampled_tokens (b, n_new), token_entropies (b, n_new) in nats,
        sequences (b, prompt_len + n_new) -- the full sequence, for downstream
        log-prob scoring with gradients enabled).
        """
        seqs = prompt_ids
        sampled = []
        entropies = []
        for _ in range(n_new_tokens):
            logits, _ = self.forward(seqs)
            last_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(last_logits, dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1, generator=generator)
            sampled.append(next_tok)
            entropies.append(ent.unsqueeze(1))
            seqs = torch.cat([seqs, next_tok], dim=1)
        return torch.cat(sampled, dim=1), torch.cat(entropies, dim=1), seqs

    def sequence_logprobs(
        self, full_seqs: torch.Tensor, prompt_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-score a full (prompt + answer) sequence WITH gradients, for the
        policy-gradient update. Returns:
          - per-answer-token log-probs (b, n_answer_tokens)
          - per-answer-token entropy (b, n_answer_tokens) in nats
          - hidden states at the answer positions (b, n_answer_tokens, d_model),
            used for the effective-rank signal.
        """
        logits, hidden = self.forward(full_seqs)
        answer_logits = logits[:, prompt_len - 1 : -1, :]  # predicts each answer token
        answer_hidden = hidden[:, prompt_len - 1 : -1, :]
        answer_tokens = full_seqs[:, prompt_len:]
        log_probs_all = F.log_softmax(answer_logits, dim=-1)
        token_logprobs = torch.gather(log_probs_all, 2, answer_tokens.unsqueeze(-1)).squeeze(-1)
        probs_all = log_probs_all.exp()
        token_entropy = -(probs_all * log_probs_all).sum(dim=-1)
        return token_logprobs, token_entropy, answer_hidden

    def clone_frozen(self) -> "TinyPolicy":
        """Return a detached, frozen deep copy for use as a KL reference."""
        ref = TinyPolicy(self.cfg)
        ref.load_state_dict(self.state_dict())
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        return ref