"""
src/models/ctc_model.py

CTC baseline model.

Pipeline:
    Image [B, 1, 64, 512]
      → ResNetEncoder → memory [B, 64, 256]
      → 2-layer BiLSTM → [B, 64, 512]
      → Linear → logits [B, 64, vocab_size]
      → CTCLoss  (training)  /  CTC greedy decode  (inference)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from src.models.cnn_encoder import ResNetEncoder


class CTCModel(nn.Module):
    """
    CNN + BiLSTM + CTC handwritten text recognition model.

    Parameters
    ----------
    vocab_size   : number of output tokens (including blank).
                   CTC blank index is assumed to be 0 (PAD token).
    d_model      : encoder feature dimension.
    lstm_hidden  : hidden size of each LSTM direction.
    lstm_layers  : number of BiLSTM layers.
    dropout      : dropout applied inside LSTM.
    pretrained   : use ImageNet weights for ResNet-18.
    """

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int   = 256,
        lstm_hidden:  int   = 256,
        lstm_layers:  int   = 2,
        dropout:      float = 0.2,
        pretrained:   bool  = True,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.blank_id    = 0     # PAD token doubles as CTC blank

        # ── Visual encoder ───────────────────────────────────────
        self.encoder = ResNetEncoder(
            d_model    = d_model,
            pretrained = pretrained,
            dropout    = dropout,
        )

        # ── BiLSTM sequence model ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = d_model,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )

        # ── CTC projection ───────────────────────────────────────
        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.ctc_head[-1].weight)
        nn.init.zeros_(self.ctc_head[-1].bias)

    # ── Forward ─────────────────────────────────────────────────

    def forward(self, images: torch.Tensor, input_widths: torch.Tensor = None):
        """
        Parameters
        ----------
        images : [B, 1, H, W]
        input_widths : [B] optionally pass scaled widths to compute true sequence lengths

        Returns
        -------
        log_probs    : [T, B, vocab_size]  log-softmax CTC output
        output_lens  : [B]  sequence length T (same for all in batch unless masked)
        """
        # Visual features: [B, T, d_model]
        memory = self.encoder(images)              # [B, T, d_model]

        # BiLSTM
        lstm_out, _ = self.lstm(memory)            # [B, T, 2*lstm_hidden]

        # CTC logits → log-probs
        logits    = self.ctc_head(lstm_out)        # [B, T, vocab_size]
        log_probs = logits.log_softmax(dim=-1)     # [B, T, vocab_size]

        # CTCLoss expects [T, B, C]
        log_probs = log_probs.permute(1, 0, 2)    # [T, B, vocab_size]

        T = log_probs.size(0)
        if input_widths is not None:
            output_lens = (input_widths // self.encoder.width_downsample).clamp(max=T).to(images.device)
        else:
            output_lens = torch.full(
                (images.size(0),), T, dtype=torch.long, device=images.device
            )

        return log_probs, output_lens

    # ── Loss ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        images:       torch.Tensor,   # [B, 1, H, W]
        labels:       torch.Tensor,   # [B, T_label]  padded label ids
        label_lens:   torch.Tensor,   # [B]           actual label lengths
        input_widths: torch.Tensor = None,  # [B]     image widths
    ) -> torch.Tensor:
        """Compute CTC loss for a batch."""
        log_probs, output_lens = self.forward(images, input_widths=input_widths)

        # CTCLoss needs labels as a 1-D concatenated tensor
        # and label_lengths as a 1-D tensor
        # CTCLoss accepts 2-D labels since PyTorch 1.9
        loss = nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)(
            log_probs,    # [T, B, C]
            labels,       # [B, T_label]  — 2-D form accepted
            output_lens,  # [B]
            label_lens,   # [B]
        )
        return loss

    # ── Greedy decode ─────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor, input_widths: torch.Tensor = None) -> list[list[int]]:
        """
        CTC greedy decoding (argmax + collapse).

        Parameters
        ----------
        images : [B, 1, H, W]
        input_widths : [B] optional true image widths before padding

        Returns
        -------
        list of token id lists, one per batch item.
        """
        log_probs, output_lens = self.forward(images, input_widths=input_widths)   # [T, B, V], [B]
        pred_ids = log_probs.argmax(dim=-1)   # [T, B]
        pred_ids = pred_ids.permute(1, 0)     # [B, T]

        results = []
        blank = self.blank_id
        for idx, seq in enumerate(pred_ids):
            # Only look at valid (non-padded) time steps
            valid_len = output_lens[idx].item()
            collapsed = []
            prev = blank
            for tok in seq[:valid_len].tolist():
                if tok != blank and tok != prev:
                    collapsed.append(tok)
                prev = tok
            results.append(collapsed)
        return results

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check ──────────────────────────────────────────────────
if __name__ == "__main__":
    model = CTCModel(vocab_size=120, pretrained=False)
    print(f"Parameters: {model.count_params():,}")

    imgs   = torch.randn(4, 1, 64, 512)
    labels = torch.randint(1, 100, (4, 12))
    llens  = torch.tensor([10, 12, 8, 11])

    loss = model.compute_loss(imgs, labels, llens)
    print(f"CTC loss: {loss.item():.4f}")

    preds = model.greedy_decode(imgs)
    print(f"Greedy decode lengths: {[len(p) for p in preds]}")
