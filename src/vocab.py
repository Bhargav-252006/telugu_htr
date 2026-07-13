"""
src/vocab.py

Telugu character vocabulary and DATA-DRIVEN transition validity matrix.

Design philosophy
─────────────────
Instead of hard-coding linguistic rules from intuition, we:
  1. Scan ALL training labels and extract every observed (prev, next) bigram.
  2. Build the validity matrix from those OBSERVED transitions only.
  3. Optionally layer on hard linguistic rules AFTER verifying they do not
     conflict with the observed data.
  4. Log a conflict report so you can audit every blocked transition.

This guarantees the constraint matrix never blocks a transition that
actually appears in the training set.

Usage
─────
  # First run (no data yet): use a permissive default vocab
  vocab = TeluguVocab()

  # After dataset is available: build from data
  vocab = TeluguVocab.from_annotation_files(["data/raw/train/labels.txt"])
  vocab.build_data_driven_matrix(["data/raw/train/labels.txt"])

  # Inspect what was learned
  vocab.print_transition_stats()

  # Save / load
  vocab.save("checkpoints/vocab.pkl")
  vocab = TeluguVocab.load("checkpoints/vocab.pkl")
"""

from __future__ import annotations

import os
import pickle
import json
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════
# Telugu Unicode character sets
# ═══════════════════════════════════════════════════════════════════

# Independent vowels  U+0C05–U+0C14
TELUGU_VOWELS: Set[str] = set(
    "\u0C05\u0C06\u0C07\u0C08\u0C09\u0C0A\u0C0B\u0C0C"
    "\u0C0E\u0C0F\u0C10\u0C12\u0C13\u0C14"
)

# Consonants  U+0C15–U+0C39, plus extras
TELUGU_CONSONANTS: Set[str] = set(
    "\u0C15\u0C16\u0C17\u0C18\u0C19"
    "\u0C1A\u0C1B\u0C1C\u0C1D\u0C1E"
    "\u0C1F\u0C20\u0C21\u0C22\u0C23"
    "\u0C24\u0C25\u0C26\u0C27\u0C28"
    "\u0C2A\u0C2B\u0C2C\u0C2D\u0C2E"
    "\u0C2F\u0C30\u0C31\u0C32\u0C33"
    "\u0C35\u0C36\u0C37\u0C38\u0C39"
    "\u0C3D\u0C58\u0C59\u0C5A"
)

# Dependent vowel signs (matras)  U+0C3E–U+0C4C, U+0C55, U+0C56
TELUGU_VOWEL_SIGNS: Set[str] = set(
    "\u0C3E\u0C3F\u0C40"
    "\u0C41\u0C42\u0C43\u0C44"
    "\u0C46\u0C47\u0C48"
    "\u0C4A\u0C4B\u0C4C"
    "\u0C55\u0C56"
)

# Virama / halant — joins consonants into conjuncts
VIRAMA: str = "\u0C4D"

# Anusvara, Visarga, Nukta
TELUGU_ANUSVARA: str = "\u0C02"
TELUGU_VISARGA:  str = "\u0C03"
TELUGU_NUKTA:    str = "\u0C00"
TELUGU_MODIFIERS: Set[str] = {TELUGU_ANUSVARA, TELUGU_VISARGA, TELUGU_NUKTA}

# Telugu digits
TELUGU_DIGITS: Set[str] = set("\u0C66\u0C67\u0C68\u0C69\u0C6A\u0C6B\u0C6C\u0C6D\u0C6E\u0C6F")

# Full set
ALL_TELUGU_CHARS: Set[str] = (
    TELUGU_VOWELS | TELUGU_CONSONANTS | TELUGU_VOWEL_SIGNS
    | {VIRAMA} | TELUGU_MODIFIERS | TELUGU_DIGITS
)

# ═══════════════════════════════════════════════════════════════════
# Special tokens
# ═══════════════════════════════════════════════════════════════════

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# ═══════════════════════════════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════════════════════════════

class TeluguVocab:
    """
    Character-level vocabulary for Telugu HTR with a data-driven
    transition validity matrix.

    The validity matrix `valid_next[i][j]` = True means token j is
    allowed as the next prediction after token i during constrained
    decoding. It is built by observing actual label bigrams.
    """

    def __init__(self, chars: Optional[List[str]] = None):
        """
        Parameters
        ----------
        chars : list of Telugu character strings to include.
                If None, uses the full predefined Telugu character set.
        """
        if chars is None:
            chars = sorted(ALL_TELUGU_CHARS)

        self._idx2char: List[str] = SPECIAL_TOKENS + chars
        self._char2idx: Dict[str, int] = {c: i for i, c in enumerate(self._idx2char)}

        self.pad_id = self._char2idx[PAD_TOKEN]
        self.sos_id = self._char2idx[SOS_TOKEN]
        self.eos_id = self._char2idx[EOS_TOKEN]
        self.unk_id = self._char2idx[UNK_TOKEN]

        # Validity matrix: initialise as fully permissive (all True)
        # Will be overwritten by build_data_driven_matrix()
        V = len(self)
        self._valid_next: List[List[bool]] = [[True] * V for _ in range(V)]

        # Statistics recorded during matrix build
        self._observed_bigrams: Dict[Tuple[int, int], int] = {}   # (prev_id, next_id) → count
        self._blocked_bigrams:  List[Tuple[int, int]]       = []  # pairs blocked by rules
        self._matrix_source: str = "permissive_default"

    # ─────────────────────────────────────────────────────────────
    # Size / lookup
    # ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._idx2char)

    def char2idx(self, ch: str) -> int:
        return self._char2idx.get(ch, self.unk_id)

    def idx2char(self, idx: int) -> str:
        if 0 <= idx < len(self._idx2char):
            return self._idx2char[idx]
        return UNK_TOKEN

    def encode(self, text: str) -> List[int]:
        return [self.char2idx(ch) for ch in text]

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        chars = []
        for i in ids:
            if strip_special and i == self.eos_id:
                break
            if strip_special and i in (self.pad_id, self.sos_id):
                continue
            chars.append(self.idx2char(i))
        return "".join(chars)

    def category(self, idx: int) -> str:
        ch = self.idx2char(idx)
        if ch == PAD_TOKEN:              return "PAD"
        if ch == SOS_TOKEN:              return "SOS"
        if ch == EOS_TOKEN:              return "EOS"
        if ch == UNK_TOKEN:              return "UNK"
        if ch in TELUGU_VOWELS:          return "VOWEL"
        if ch in TELUGU_CONSONANTS:      return "CONSONANT"
        if ch in TELUGU_VOWEL_SIGNS:     return "VOWEL_SIGN"
        if ch == VIRAMA:                 return "VIRAMA"
        if ch in TELUGU_MODIFIERS:       return "MODIFIER"
        if ch in TELUGU_DIGITS:          return "DIGIT"
        return "OTHER"

    # ─────────────────────────────────────────────────────────────
    # Data-driven transition matrix
    # ─────────────────────────────────────────────────────────────

    def build_data_driven_matrix(
        self,
        annotation_files: List[str],
        soft_mode: bool = False,
        soft_penalty: float = -10.0,
        verbose: bool = True,
    ):
        """
        Build the transition validity matrix from OBSERVED label bigrams.

        Algorithm
        ─────────
        1. Parse every label in the annotation files.
        2. For each label, generate synthetic bigrams:
             SOS → first_char
             char[i] → char[i+1]   for all consecutive pairs
             last_char → EOS
        3. Record all observed (prev_id, next_id) pairs.
        4. Set valid_next[i][j] = True  iff  (i,j) was observed in data.
           Special cases:
             - PAD → nothing  (always blocked)
             - EOS → nothing  (always blocked)
             - UNK → everything  (always allowed, it's a fallback)
             - everything → UNK  (always allowed)
        5. Store the set of blocked pairs for audit.

        Parameters
        ----------
        annotation_files : list of annotation .txt paths (train only — DO NOT
                           include val/test to avoid data leakage).
        soft_mode        : if True, don't fully block — return a float penalty
                           tensor instead (useful if matrix is exported to GPU).
        verbose          : print audit report.
        """
        V = len(self)
        observed: Dict[Tuple[int, int], int] = defaultdict(int)

        total_labels = 0
        for path in annotation_files:
            if not os.path.exists(path):
                print(f"[Vocab] WARNING: annotation file not found: {path}")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    label = unicodedata.normalize("NFC", parts[1].strip())
                    if not label:
                        continue

                    total_labels += 1
                    ids = self.encode(label)

                    # SOS → first token
                    if ids:
                        observed[(self.sos_id, ids[0])] += 1

                    # Consecutive pairs
                    for p, n in zip(ids, ids[1:]):
                        observed[(p, n)] += 1

                    # Last token → EOS
                    if ids:
                        observed[(ids[-1], self.eos_id)] += 1

        if verbose:
            print(f"\n[Vocab] Scanned {total_labels:,} labels from {len(annotation_files)} file(s).")
            print(f"[Vocab] Observed {len(observed):,} unique (prev, next) token bigrams.")

        # ── Build validity matrix ──────────────────────────────
        # Default: block everything, then allow what was observed
        valid = [[False] * V for _ in range(V)]

        for (prev_id, next_id), count in observed.items():
            if 0 <= prev_id < V and 0 <= next_id < V:
                valid[prev_id][next_id] = True

        # Always allow transitions to/from UNK (safety fallback)
        for i in range(V):
            valid[i][self.unk_id] = True
            valid[self.unk_id][i] = True

        # Always block PAD and EOS as previous tokens
        for j in range(V):
            valid[self.pad_id][j] = False
            valid[self.eos_id][j] = False

        # PAD as next is always blocked (PAD only pads sequences, never predicted)
        for i in range(V):
            valid[i][self.pad_id] = False

        self._valid_next       = valid
        self._observed_bigrams = dict(observed)
        self._matrix_source    = "data_driven"

        if verbose:
            self._audit_report(V)

    def _audit_report(self, V: int):
        """Print a human-readable summary of the transition matrix."""
        total_pairs   = V * V
        allowed_pairs = sum(self._valid_next[i][j] for i in range(V) for j in range(V))
        blocked_pairs = total_pairs - allowed_pairs
        coverage_pct  = 100.0 * allowed_pairs / total_pairs

        print(f"\n[Vocab] ── Transition Matrix Audit ─────────────────────────")
        print(f"  Vocabulary size   : {V}")
        print(f"  Total token pairs : {total_pairs:,}")
        print(f"  Allowed pairs     : {allowed_pairs:,}  ({coverage_pct:.1f}%)")
        print(f"  Blocked pairs     : {blocked_pairs:,}  ({100-coverage_pct:.1f}%)")
        print(f"  Matrix source     : {self._matrix_source}")

        # Per-category summary
        cats = set(self.category(i) for i in range(V))
        print(f"\n  Allowed transitions by (prev_category → next_category):")
        cat_list = ["SOS", "VOWEL", "CONSONANT", "VOWEL_SIGN", "VIRAMA",
                    "MODIFIER", "DIGIT", "EOS", "UNK"]
        header = f"  {'PREV':>12} |" + "".join(f"{c:>12}" for c in cat_list)
        print(header)
        print("  " + "-" * (13 + 12 * len(cat_list)))

        for prev_cat in cat_list:
            prev_ids = [i for i in range(V) if self.category(i) == prev_cat]
            row = f"  {prev_cat:>12} |"
            for nxt_cat in cat_list:
                nxt_ids = [j for j in range(V) if self.category(j) == nxt_cat]
                allowed = sum(
                    1 for p in prev_ids for n in nxt_ids if self._valid_next[p][n]
                )
                total = len(prev_ids) * len(nxt_ids)
                if total == 0:
                    row += f"{'—':>12}"
                else:
                    row += f"{allowed:>5}/{total:<6}"
            print(row)
        print()

    # ─────────────────────────────────────────────────────────────
    # Constrained decoding interface
    # ─────────────────────────────────────────────────────────────

    def get_valid_next_mask(self, prev_token_id: int) -> List[bool]:
        """
        Return a boolean list of length V.
        True  → token at that index is valid after prev_token_id.
        False → blocked.
        """
        if 0 <= prev_token_id < len(self._valid_next):
            return self._valid_next[prev_token_id]
        return [True] * len(self)   # fallback: allow everything

    def get_valid_next_tensor(
        self, prev_token_id: int, device: str = "cpu"
    ):
        """
        Return a boolean torch.Tensor of shape [V] on the given device.
        Useful for direct logit masking in the decoder.
        """
        import torch
        mask = self._valid_next[prev_token_id]
        return torch.tensor(mask, dtype=torch.bool, device=device)

    def is_valid_transition(self, prev_id: int, next_id: int) -> bool:
        return bool(self._valid_next[prev_id][next_id])

    # ─────────────────────────────────────────────────────────────
    # Validation helper — run BEFORE training to catch rule issues
    # ─────────────────────────────────────────────────────────────

    def validate_against_split(
        self, annotation_file: str, split_name: str = "val"
    ) -> Dict:
        """
        Run through a split (val or test) and count how many label
        transitions are blocked by the current matrix.

        A blocked val/test transition means the constraint will HURT
        decoding on that sample. This number should be 0 or very close to 0.

        Returns a dict with violation statistics.
        """
        total_transitions  = 0
        blocked_count      = 0
        blocked_examples   = []   # (label, prev_char, next_char)

        with open(annotation_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                label = unicodedata.normalize("NFC", parts[1].strip())
                ids   = self.encode(label)
                full  = [self.sos_id] + ids + [self.eos_id]

                for p, n in zip(full, full[1:]):
                    total_transitions += 1
                    if not self._valid_next[p][n]:
                        blocked_count += 1
                        if len(blocked_examples) < 20:
                            blocked_examples.append(
                                (label, self.idx2char(p), self.idx2char(n))
                            )

        violation_rate = blocked_count / max(total_transitions, 1)
        print(f"\n[Vocab] Validation against [{split_name}]:")
        print(f"  Total transitions   : {total_transitions:,}")
        print(f"  Blocked transitions : {blocked_count:,}  ({violation_rate*100:.3f}%)")

        if blocked_examples:
            print(f"  First blocked examples:")
            for label, prev_ch, next_ch in blocked_examples[:5]:
                print(f"    label='{label}'  prev='{prev_ch}'  next='{next_ch}'")

        if violation_rate > 0.001:
            print(f"  ⚠️  WARNING: >0.1% transitions blocked on {split_name}.")
            print(f"     Consider expanding training data coverage or relaxing rules.")
        else:
            print(f"  ✓  Constraint matrix is safe on {split_name}.")

        return {
            "split": split_name,
            "total_transitions": total_transitions,
            "blocked_count": blocked_count,
            "violation_rate": violation_rate,
            "blocked_examples": blocked_examples,
        }

    # ─────────────────────────────────────────────────────────────
    # Factory: build from annotation files
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def from_annotation_files(
        cls, annotation_files: List[str], build_matrix: bool = True
    ) -> "TeluguVocab":
        """
        1. Scan labels to collect observed characters.
        2. Build vocab from those characters only.
        3. Optionally build data-driven transition matrix from train labels.

        Parameters
        ----------
        annotation_files : TRAIN split annotation files only.
        build_matrix     : if True, also call build_data_driven_matrix().
        """
        from collections import Counter
        char_counter: Counter = Counter()

        for path in annotation_files:
            if not os.path.exists(path):
                print(f"[Vocab] WARNING: {path} not found — skipping.")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        char_counter.update(unicodedata.normalize("NFC", parts[1]))

        # Keep only chars that are actual Telugu (filter noise)
        chars = sorted(ch for ch in char_counter if ch in ALL_TELUGU_CHARS)
        # Include any extra unknown chars (handle dirty labels gracefully)
        extra = sorted(ch for ch in char_counter
                       if ch not in ALL_TELUGU_CHARS and ch not in SPECIAL_TOKENS)
        if extra:
            print(f"[Vocab] Found {len(extra)} non-Telugu chars in labels: {extra[:10]}")

        vocab = cls(chars + extra)
        print(f"[Vocab] Built vocabulary: {len(vocab)} tokens "
              f"({len(chars)} Telugu + {len(extra)} extra + {len(SPECIAL_TOKENS)} special)")

        if build_matrix:
            vocab.build_data_driven_matrix(annotation_files)

        return vocab

    # ─────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Vocab] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TeluguVocab":
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        print(f"[Vocab] Loaded from {path} (size={len(vocab)})")
        return vocab

    def __repr__(self) -> str:
        return (
            f"TeluguVocab(size={len(self)}, "
            f"matrix_source='{self._matrix_source}')"
        )


# ═══════════════════════════════════════════════════════════════════
# Module-level default vocab (permissive, no matrix built yet)
# ═══════════════════════════════════════════════════════════════════
DEFAULT_VOCAB = TeluguVocab()


# ═══════════════════════════════════════════════════════════════════
# CLI entry point for building & saving vocab from dataset
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.vocab <train_annotation.txt> <output_vocab.pkl> [val_annotation.txt]")
        sys.exit(1)

    train_ann  = sys.argv[1]
    output_pkl = sys.argv[2]
    val_ann    = sys.argv[3] if len(sys.argv) > 3 else None

    # Build from training data
    vocab = TeluguVocab.from_annotation_files([train_ann], build_matrix=True)

    # Validate against val split (should be 0 violations)
    if val_ann and os.path.exists(val_ann):
        vocab.validate_against_split(val_ann, split_name="val")

    # Save
    vocab.save(output_pkl)

    # Quick round-trip test
    word = next(
        (parts[1] for line in open(train_ann, encoding="utf-8")
         for parts in [line.strip().split(maxsplit=1)] if len(parts) == 2),
        "కాలం"
    )
    ids  = vocab.encode(word)
    back = vocab.decode(ids)
    print(f"\nRound-trip: '{word}' → {ids} → '{back}'")
    print(f"\nDone. Vocab saved to: {output_pkl}")
