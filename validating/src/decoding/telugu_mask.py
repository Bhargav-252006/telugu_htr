"""
src/decoding/telugu_mask.py

Utilities for applying the data-driven Telugu script constraint mask
during autoregressive decoding.

The core object is the TeluguMask, a thin wrapper around the validity
matrix stored in TeluguVocab. It provides:
  - apply_to_logits()  — zero-out illegal next tokens before argmax / beam
  - batch_apply()      — vectorised apply for whole batches
  - stats_report()     — how often the mask actually fires during a decode run
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class MaskStats:
    """Accumulated statistics about how often the mask fires."""
    total_steps:   int = 0
    masked_steps:  int = 0      # steps where ≥1 token was blocked
    tokens_masked: int = 0      # total number of tokens blocked across all steps

    def update(self, blocked_count: int):
        self.total_steps   += 1
        if blocked_count > 0:
            self.masked_steps  += 1
            self.tokens_masked += blocked_count

    def report(self) -> str:
        if self.total_steps == 0:
            return "MaskStats: no steps recorded."
        fire_rate = 100.0 * self.masked_steps / self.total_steps
        return (
            f"MaskStats | steps={self.total_steps} | "
            f"mask_fired={self.masked_steps} ({fire_rate:.1f}%) | "
            f"tokens_blocked={self.tokens_masked}"
        )


class TeluguMask:
    """
    Applies the transition validity matrix from a TeluguVocab to decoder
    logits during autoregressive inference.

    Usage
    ─────
        mask_module = TeluguMask(vocab)

        # Inside decode loop:
        logits = model.decode_step(memory, prev_tokens)
        logits = mask_module.apply_to_logits(logits, prev_token_ids)
        next_tokens = logits.argmax(-1)
    """

    NEG_INF = float("-inf")

    def __init__(self, vocab, collect_stats: bool = True):
        """
        Parameters
        ----------
        vocab        : TeluguVocab instance (must have _valid_next built).
        collect_stats: if True, accumulate firing statistics for analysis.
        """
        self.vocab = vocab
        self.V     = len(vocab)
        self.stats = MaskStats() if collect_stats else None

        # Pre-cache the validity matrix as a GPU-friendly boolean tensor
        # Shape: [V, V]  valid_tensor[i, j] = True if j is valid after i
        valid_list = vocab._valid_next
        self._valid_tensor: Optional[torch.Tensor] = torch.tensor(
            valid_list, dtype=torch.bool
        )   # will be moved to device on first use

    def _ensure_device(self, device: torch.device):
        if self._valid_tensor.device != device:
            self._valid_tensor = self._valid_tensor.to(device)

    # ── Single-step apply ─────────────────────────────────────────

    def apply_to_logits(
        self,
        logits:          torch.Tensor,   # [B, V]
        prev_token_ids:  torch.Tensor,   # [B]   long
    ) -> torch.Tensor:
        """
        Mask illegal next-token logits to -inf.

        Parameters
        ----------
        logits         : raw decoder logits, shape [B, V].
        prev_token_ids : the previously predicted token for each item, [B].

        Returns
        -------
        Masked logits [B, V].  The original tensor is modified in-place
        for efficiency and also returned.
        """
        device = logits.device
        self._ensure_device(device)

        B = logits.size(0)
        for b in range(B):
            prev = prev_token_ids[b].item()
            valid_row = self._valid_tensor[prev]         # [V]  bool
            blocked   = ~valid_row                       # [V]  True = block
            n_blocked = blocked.sum().item()

            logits[b][blocked] = self.NEG_INF

            if self.stats is not None:
                self.stats.update(n_blocked)

        return logits

    # ── Vectorised batch apply ────────────────────────────────────

    def apply_to_logits_vectorised(
        self,
        logits:         torch.Tensor,   # [B, V]
        prev_token_ids: torch.Tensor,   # [B]   long
    ) -> torch.Tensor:
        """
        Fully vectorised version using advanced indexing.
        Faster for large batches / large vocabularies.
        """
        device = logits.device
        self._ensure_device(device)

        # Gather validity rows for all prev tokens in batch: [B, V]
        prev   = prev_token_ids.clamp(0, self.V - 1)    # safety
        valid  = self._valid_tensor[prev]                # [B, V]
        logits = logits.masked_fill(~valid, self.NEG_INF)

        if self.stats is not None:
            n_blocked = (~valid).sum().item()
            self.stats.update(n_blocked)

        return logits

    # ── Beam search version ───────────────────────────────────────

    def apply_to_beam_logits(
        self,
        logits:         torch.Tensor,   # [K, V]  K = beam_size
        prev_token_ids: List[int],
    ) -> torch.Tensor:
        """
        Apply mask to K beams, each with its own previous token.
        """
        device = logits.device
        self._ensure_device(device)

        for k, prev in enumerate(prev_token_ids):
            prev = max(0, min(prev, self.V - 1))
            valid_row = self._valid_tensor[prev]
            logits[k][~valid_row] = self.NEG_INF

        return logits

    # ── Statistics ───────────────────────────────────────────────

    def reset_stats(self):
        if self.stats is not None:
            self.stats = MaskStats()

    def print_stats(self):
        if self.stats is not None:
            print(f"[TeluguMask] {self.stats.report()}")
        else:
            print("[TeluguMask] Stats collection disabled.")
