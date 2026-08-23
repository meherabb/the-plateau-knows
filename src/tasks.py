"""
Synthetic algorithmic task families for RL-grokking experiments.

Design rationale (documented here so it travels with the code, and is
restated in the notebook's markdown):

  - Both tasks are defined over (a, b) pairs drawn from a FIXED, finite,
    seeded 97x97 = 9,409-pair space, with a fixed train/test split -- this
    now matches the classic supervised-grokking setup (Power et al., 2022)
    on purpose, not just in spirit. An earlier version of Task A used three
    independent inputs (a, b, c), which blows the space up to 97^3 ~= 913K --
    a ~97x larger space that the same step*batch exposure covers only ~1%
    as thoroughly. That mismatch was diagnosed directly: a 12,000- and then
    30,000-step real sweep showed reward flat at the random baseline in
    BOTH the dense and binary phases, and a supervised (non-RL) control
    on the 3-input version showed the same flatness -- i.e. it wasn't
    even a training-signal-type problem, it was an exposure problem. The
    two-input, fixed-split design below is deliberately the same scale the
    literature this project builds on has already validated converges in a
    tractable step budget.
  - The train/test split itself is fixed by a task-level seed shared across
    all training seeds/schedules -- every run sees the identical set of
    memorizable pairs; only model initialization and sampling stochasticity
    vary by run seed. This isolates "does training dynamics affect
    breakthrough timing" from "did this run happen to get an easier data
    split," which is the property grokking studies need from a train/test
    split.
  - TASK A (single-token): y = (a * b) mod 97, a random policy succeeds
    with probability 1/97 (~1.03%) per attempt.
  - TASK B (two-token): d1 = (a + b) mod 97, d2 = (a * b) mod P_SUB -- a
    genuinely different composition (different operations, an added
    modulus) over the SAME (a, b) input space, so the two task families are
    comparable in scale while remaining structurally distinct. Requires two
    dependent correct predictions, so a random policy succeeds with
    probability (1/97) * (1/P_SUB); this is the only place actual
    autoregressive (length > 1) generation is exercised.

Both tasks expose the same interface: a Task object that can (a) sample a
batch of prompts from a given split, (b) score a batch of candidate answers
against a dense (graded) or binary reward, and (c) render prompts/answers to
token ids for the model.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import torch

RewardMode = Literal["dense", "binary"]
Split = Literal["train", "test"]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
P_MAIN = 97   # matches Power et al. 2022's classic modulus
P_SUB = 23    # second modulus used only by Task B's second digit

PAD, BOS, PLUS, TIMES, EQUALS = range(P_MAIN, P_MAIN + 5)
VOCAB_SIZE = P_MAIN + 5  # special tokens fit inside the same table as P_MAIN numerals

TRAIN_TEST_SPLIT_SEED = 1234    # fixed across all runs -- see module docstring
DEFAULT_TRAIN_FRACTION = 0.5    # matches common practice in the grokking literature


def circular_distance(pred: torch.Tensor, target: torch.Tensor, modulus: int) -> torch.Tensor:
    """Minimal distance between pred and target on a ring of size `modulus`."""
    diff = (pred - target) % modulus
    return torch.minimum(diff, modulus - diff)


@dataclasses.dataclass
class TaskBatch:
    """A batch of prompts plus everything needed to score answers to them."""
    prompt_ids: torch.Tensor       # (batch, prompt_len) long
    targets: torch.Tensor          # (batch, n_answer_tokens) long -- ground truth
    n_answer_tokens: int
    answer_modulus: list  # modulus for each answer token position, e.g. [97] or [97, 23]


def _fixed_pair_split(train_fraction: float = DEFAULT_TRAIN_FRACTION):
    """All (a, b) in [0, P_MAIN) x [0, P_MAIN), shuffled once under a fixed
    seed and split into train/test index arrays. Shared by both tasks so
    they're defined over literally the same underlying pair space."""
    rng = np.random.default_rng(TRAIN_TEST_SPLIT_SEED)
    all_a, all_b = np.meshgrid(np.arange(P_MAIN), np.arange(P_MAIN), indexing="ij")
    all_a, all_b = all_a.ravel(), all_b.ravel()
    order = rng.permutation(len(all_a))
    n_train = int(train_fraction * len(all_a))
    train_idx, test_idx = order[:n_train], order[n_train:]
    return all_a, all_b, train_idx, test_idx


class _FixedPairTask:
    """Shared machinery: sampling (with replacement) from a fixed train or
    test index set of (a, b) pairs. Subclasses define the prompt rendering
    and the target function(s)."""

    def __init__(self, train_fraction: float = DEFAULT_TRAIN_FRACTION):
        self._all_a, self._all_b, self.train_idx, self.test_idx = _fixed_pair_split(train_fraction)

    def _sample_pairs(self, batch_size: int, rng: np.random.Generator, split: Split):
        pool = self.train_idx if split == "train" else self.test_idx
        chosen = rng.choice(pool, size=batch_size, replace=True)
        return self._all_a[chosen], self._all_b[chosen]


class TaskA(_FixedPairTask):
    """y = (a * b) mod P_MAIN, single-token answer, fixed finite pair pool."""

    name = "task_a_single_token"
    n_answer_tokens = 1
    answer_modulus = [P_MAIN]
    prompt_len = 5  # BOS a TIMES b EQUALS

    def sample_batch(self, batch_size: int, rng: np.random.Generator, split: Split = "train") -> TaskBatch:
        a, b = self._sample_pairs(batch_size, rng, split)
        y = (a * b) % P_MAIN

        prompt = np.stack(
            [np.full(batch_size, BOS), a, np.full(batch_size, TIMES), b, np.full(batch_size, EQUALS)],
            axis=1,
        )
        return TaskBatch(
            prompt_ids=torch.as_tensor(prompt, dtype=torch.long),
            targets=torch.as_tensor(y[:, None], dtype=torch.long),
            n_answer_tokens=1,
            answer_modulus=[P_MAIN],
        )

    def reward(self, targets: torch.Tensor, predictions: torch.Tensor, mode: RewardMode) -> torch.Tensor:
        """targets, predictions: (batch, 1). Returns (batch,) reward in [0, 1]."""
        dist = circular_distance(predictions[:, 0], targets[:, 0], P_MAIN).float()
        if mode == "binary":
            return (dist == 0).float()
        return torch.clamp(1.0 - dist / (P_MAIN / 2), min=0.0)


class TaskB(_FixedPairTask):
    """
    Two dependent digits over the SAME (a, b) pair pool as Task A:
    d1 = (a + b) mod P_MAIN, d2 = (a * b) mod P_SUB. Structurally different
    composition (different operations, an added modulus) at comparable
    input-space scale, and the only place real length>1 autoregressive
    sampling is exercised.
    """

    name = "task_b_two_token"
    n_answer_tokens = 2
    answer_modulus = [P_MAIN, P_SUB]
    prompt_len = 5

    def sample_batch(self, batch_size: int, rng: np.random.Generator, split: Split = "train") -> TaskBatch:
        a, b = self._sample_pairs(batch_size, rng, split)
        d1 = (a + b) % P_MAIN
        d2 = (a * b) % P_SUB

        prompt = np.stack(
            [np.full(batch_size, BOS), a, np.full(batch_size, PLUS), b, np.full(batch_size, EQUALS)],
            axis=1,
        )
        targets = np.stack([d1, d2], axis=1)
        return TaskBatch(
            prompt_ids=torch.as_tensor(prompt, dtype=torch.long),
            targets=torch.as_tensor(targets, dtype=torch.long),
            n_answer_tokens=2,
            answer_modulus=[P_MAIN, P_SUB],
        )

    def reward(self, targets: torch.Tensor, predictions: torch.Tensor, mode: RewardMode) -> torch.Tensor:
        """targets, predictions: (batch, 2). Returns (batch,) reward in [0, 1]."""
        d1 = circular_distance(predictions[:, 0], targets[:, 0], P_MAIN).float()
        d2 = circular_distance(predictions[:, 1], targets[:, 1], P_SUB).float()
        if mode == "binary":
            return ((d1 == 0) & (d2 == 0)).float()
        credit1 = torch.clamp(1.0 - d1 / (P_MAIN / 2), min=0.0)
        credit2 = torch.clamp(1.0 - d2 / (P_SUB / 2), min=0.0)
        return 0.5 * (credit1 + credit2)


TASK_REGISTRY = {"task_a_single_token": TaskA, "task_b_two_token": TaskB}


def get_task(name: str):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Options: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()