"""
src/models/ar_model.py

Autoregressive Transformer decoder model.

Pipeline:
    Image [B, 1, 64, 512]
      → ResNetEncoder  → encoder memory [B, 64, 256]
      → Transformer Decoder (cross-attention over memory)
         Input: shifted label tokens (teacher forcing during training)
      → Linear → logits [B, T, vocab_size]
      → Cross-Entropy loss  (training)
      → Greedy / beam decode  (inference)

The decoder is causal: each position can only attend to previous positions
(standard auto-regressive constraint via causal mask).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_encoder import ResNetEncoder, PositionalEncoding


class ARModel(nn.Module):
    """
    CNN Encoder + Autoregressive Transformer Decoder for Telugu HTR.

    Parameters
    ----------
    vocab_size        : size of the character vocabulary.
    sos_id / eos_id   : special token ids.
    d_model           : feature dimension (encoder and decoder share this).
    nhead             : number of attention heads.
    num_decoder_layers: number of Transformer decoder layers.
    dim_feedforward   : inner dimension of the decoder FFN.
    dropout           : dropout throughout.
    max_label_len     : maximum label sequence length (for positional enc).
    label_smoothing   : label smoothing for cross-entropy loss.
    pretrained        : use ImageNet weights for ResNet-18.
    """

    def __init__(
        self,
        vocab_size:         int,
        sos_id:             int   = 1,
        eos_id:             int   = 2,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_encoder_layers: int   = 2,
        num_decoder_layers: int   = 4,
        dim_feedforward:    int   = 1024,
        dropout:            float = 0.1,
        max_label_len:      int   = 64,
        label_smoothing:    float = 0.1,
        pretrained:         bool  = True,
        high_res_temporal:  bool  = False,
        ctc_weight:         float = 0.3,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.sos_id      = sos_id
        self.eos_id      = eos_id
        self.d_model     = d_model
        self.max_label_len = max_label_len

        # ── Visual encoder ───────────────────────────────────────
        self.cnn_encoder = ResNetEncoder(
            d_model    = d_model,
            pretrained = pretrained,
            dropout    = dropout,
            high_res_temporal = high_res_temporal,
        )

        if num_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        else:
            self.transformer_encoder = None

        # ── Token embedding + positional encoding (decoder side) ─
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dec_pos_enc = PositionalEncoding(d_model, max_len=max_label_len + 2, dropout=dropout)

        # ── Transformer decoder ──────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable training
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers = num_decoder_layers,
        )

        # ── Output projection ────────────────────────────────────
        self.output_proj = nn.Linear(d_model, vocab_size)

        # ── Auxiliary CTC head for joint training ────────────────
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.ctc_weight = ctc_weight  # weighting factor for CTC loss (0 = pure CE)

        # ── Loss Function ─────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            ignore_index    = 0,    # PAD
            label_smoothing = label_smoothing,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ── Causal mask ──────────────────────────────────────────────

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for autoregressive decoding. [sz, sz]"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask    # True = blocked

    # ── Padding mask ─────────────────────────────────────────────

    def _pad_mask(self, seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Key padding mask: True where token == pad_id. [B, T]"""
        return seq == pad_id

    def _encoder_padding_mask(self, input_widths: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Create memory_key_padding_mask. True = padded position to ignore. [B, S]"""
        encoder_lens = (input_widths // self.cnn_encoder.width_downsample).clamp(max=seq_len)
        positions = torch.arange(seq_len, device=input_widths.device).unsqueeze(0)  # [1, S]
        mask = positions >= encoder_lens.unsqueeze(1)  # [B, S]
        return mask

    # ── Forward (teacher forcing) ────────────────────────────────

    def forward(
        self,
        images:  torch.Tensor,   # [B, 1, H, W]
        tgt_ids: torch.Tensor,   # [B, T]  shifted input  (SOS + label[:-1])
        input_widths: torch.Tensor = None, # [B] optional image widths
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Returns
        -------
        logits : [B, T, vocab_size]
        """
        B, T = tgt_ids.shape
        device = images.device

        # Encoder memory
        memory = self.cnn_encoder(images)              # [B, S, d_model]

        memory_pad_mask = None
        if input_widths is not None:
            memory_pad_mask = self._encoder_padding_mask(input_widths, memory.size(1))

        if self.transformer_encoder is not None:
            memory = self.transformer_encoder(
                src=memory, 
                src_key_padding_mask=memory_pad_mask
            )

        # Decoder input embeddings + positional encoding
        tgt_emb = self.token_embed(tgt_ids)        # [B, T, d_model]
        tgt_emb = self.dec_pos_enc(tgt_emb)

        # Causal mask
        causal_mask = self._causal_mask(T, device)

        # Key padding mask for decoder input
        tgt_key_pad = self._pad_mask(tgt_ids)      # [B, T]

        # Transformer decoder
        dec_out = self.decoder(
            tgt             = tgt_emb,
            memory          = memory,
            tgt_mask        = causal_mask,
            tgt_key_padding_mask = tgt_key_pad,
            memory_key_padding_mask = memory_pad_mask,
        )                                          # [B, T, d_model]

        logits = self.output_proj(dec_out)         # [B, T, vocab_size]
        return logits

    # ── Loss ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        images:     torch.Tensor,   # [B, 1, H, W]
        labels:     torch.Tensor,   # [B, T+2]  SOS + label + EOS (padded)
        label_lens: torch.Tensor,   # [B]  lengths including SOS and EOS
        input_widths: torch.Tensor = None, # [B]
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with teacher forcing.

        labels should be: [SOS, c1, c2, ..., cn, EOS, PAD, PAD, ...]
        Decoder input  : labels[:, :-1]  → [SOS, c1, ..., cn]
        Target         : labels[:, 1:]   → [c1, ..., cn, EOS]
        """
        decoder_input = labels[:, :-1]    # [B, T]
        target        = labels[:, 1:]     # [B, T]

        # 1. Run Encoder
        memory = self.cnn_encoder(images)
        memory_pad_mask = None
        if input_widths is not None:
            memory_pad_mask = self._encoder_padding_mask(input_widths, memory.size(1))
        if self.transformer_encoder is not None:
            memory = self.transformer_encoder(src=memory, src_key_padding_mask=memory_pad_mask)

        # 2. Run Decoder for AR CE Loss
        tgt_emb = self.token_embed(decoder_input)
        tgt_emb = self.dec_pos_enc(tgt_emb)
        causal_mask = self._causal_mask(decoder_input.size(1), images.device)
        tgt_key_pad = self._pad_mask(decoder_input)
        dec_out = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_pad, memory_key_padding_mask=memory_pad_mask
        )
        logits = self.output_proj(dec_out)
        ce_loss = self.criterion(logits.reshape(-1, self.vocab_size), target.reshape(-1))

        # 3. Auxiliary CTC Loss on Encoder Memory (skip if weight == 0)
        if self.ctc_weight > 0:
            ctc_logits = self.ctc_head(memory)
            log_probs = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)  # [T, B, C]
            
            batch_size = labels.size(0)
            ctc_targets = []
            ctc_label_lens = []
            for i in range(batch_size):
                llen = label_lens[i].item()
                # extract actual tokens without SOS (pos 0) and EOS (pos llen-1)
                valid_tokens = labels[i, 1:llen-1]
                ctc_targets.append(valid_tokens)
                ctc_label_lens.append(len(valid_tokens))
                
            max_len = max(ctc_label_lens) if ctc_label_lens else 0
            ctc_target_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=labels.device)
            for i in range(batch_size):
                ctc_target_tensor[i, :ctc_label_lens[i]] = ctc_targets[i]
                
            T_ctc = log_probs.size(0)
            if input_widths is not None:
                output_lens = (input_widths // self.cnn_encoder.width_downsample).clamp(max=T_ctc).to(images.device)
            else:
                output_lens = torch.full((batch_size,), T_ctc, dtype=torch.long, device=images.device)
                
            ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
            ctc_loss = ctc_loss_fn(
                log_probs, ctc_target_tensor, output_lens, 
                torch.tensor(ctc_label_lens, dtype=torch.long, device=images.device)
            )

            # Combine losses
            total_loss = (1.0 - self.ctc_weight) * ce_loss + self.ctc_weight * ctc_loss
            return total_loss, ce_loss.detach(), ctc_loss.detach()
        else:
            # Pure CE — no auxiliary CTC
            return ce_loss, ce_loss.detach(), torch.tensor(0.0, device=images.device)

    # ── Greedy decode (inference) ─────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        images:    torch.Tensor,              # [B, 1, H, W]
        max_len:   int           = 32,
        vocab     = None,                     # TeluguVocab — for constrained decoding
        constrain: bool          = False,     # apply validity matrix
        constrain_penalty: float = None,      # soft penalty for invalid transitions
        input_widths: torch.Tensor = None,    # [B]
    ) -> list[list[int]]:
        """
        Autoregressive greedy decoding.

        If constrain=True and vocab is provided, logits for illegal
        next tokens are set to -inf before argmax.

        Returns
        -------
        list of token id lists (without SOS, stopped at EOS).
        """
        B      = images.size(0)
        device = images.device

        memory = self.cnn_encoder(images)             # [B, S, d_model]
        
        memory_pad_mask = None
        if input_widths is not None:
            memory_pad_mask = self._encoder_padding_mask(input_widths, memory.size(1))

        if self.transformer_encoder is not None:
            memory = self.transformer_encoder(
                src=memory, 
                src_key_padding_mask=memory_pad_mask
            )

        # Start with SOS token
        generated = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            tgt_emb   = self.token_embed(generated)          # [B, t, d_model]
            tgt_emb   = self.dec_pos_enc(tgt_emb)
            causal_mk = self._causal_mask(generated.size(1), device)

            dec_out = self.decoder(
                tgt    = tgt_emb,
                memory = memory,
                tgt_mask = causal_mk,
                memory_key_padding_mask = memory_pad_mask,
            )                                                # [B, t, d_model]

            logits = self.output_proj(dec_out[:, -1, :])    # [B, V]

            # ── Telugu constraint ──────────────────────────────
            if constrain and vocab is not None:
                prev_tokens = generated[:, -1]               # [B]
                for b in range(B):
                    if finished[b]:
                        continue
                    prev_id = prev_tokens[b].item()
                    valid_mask = vocab.get_valid_next_tensor(prev_id, device=device)
                    if constrain_penalty is not None:
                        logits[b] = torch.where(valid_mask, logits[b], logits[b] - constrain_penalty)
                    else:
                        logits[b][~valid_mask] = float("-inf")

            next_tok = logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            # Mark finished sequences
            finished = finished | (next_tok.squeeze(1) == self.eos_id)

            generated = torch.cat([generated, next_tok], dim=1)

            if finished.all():
                break

        # Strip SOS, stop at EOS
        results = []
        for row in generated:
            ids = row.tolist()[1:]   # remove SOS
            out = []
            for tok in ids:
                if tok == self.eos_id:
                    break
                out.append(tok)
            results.append(out)

        return results

    # ── Beam search decode ────────────────────────────────────────

    @torch.no_grad()
    def beam_decode(
        self,
        images:    torch.Tensor,
        beam_size: int  = 5,
        max_len:   int  = 32,
        vocab      = None,
        constrain: bool = False,
        constrain_penalty: float = None,
        input_widths: torch.Tensor = None,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        """
        Beam search decoding — processes one image at a time for simplicity.

        Returns list of best token id sequences (without SOS/EOS).
        """
        B      = images.size(0)
        device = images.device
        results = []

        for b in range(B):
            img    = images[b:b+1]                           # [1, 1, H, W]
            memory = self.cnn_encoder(img)                       # [1, S, d_model]
            
            memory_pad_mask = None
            if input_widths is not None:
                memory_pad_mask = self._encoder_padding_mask(input_widths[b:b+1], memory.size(1))
            
            if self.transformer_encoder is not None:
                memory = self.transformer_encoder(src=memory, src_key_padding_mask=memory_pad_mask)

            if memory_pad_mask is not None:
                memory_pad_mask = memory_pad_mask.expand(beam_size, -1)

            memory = memory.expand(beam_size, -1, -1)       # [K, S, d_model]

            # beams: list of (score, token_ids)
            beams  = [(0.0, [self.sos_id])]
            done   = []

            for step in range(max_len):
                all_candidates = []

                # Expand memory to current beam count
                cur_k  = len(beams)
                mem_k  = memory[:cur_k]

                # Build current token sequences
                seqs = torch.tensor(
                    [b_tok for _, b_tok in beams],
                    dtype=torch.long, device=device
                )                                            # [cur_k, t]

                tgt_emb   = self.token_embed(seqs)
                tgt_emb   = self.dec_pos_enc(tgt_emb)
                causal_mk = self._causal_mask(seqs.size(1), device)

                dec_out = self.decoder(
                    tgt    = tgt_emb,
                    memory = mem_k,
                    tgt_mask = causal_mk,
                    memory_key_padding_mask = memory_pad_mask[:cur_k] if memory_pad_mask is not None else None,
                )                                            # [cur_k, t, d_model]

                logits = self.output_proj(dec_out[:, -1, :])  # [cur_k, V]

                # Apply Telugu constraints per beam
                if constrain and vocab is not None:
                    for k_i, (_, tok_seq) in enumerate(beams):
                        prev_id = tok_seq[-1]
                        valid_mask = vocab.get_valid_next_tensor(prev_id, device=device)
                        if constrain_penalty is not None:
                            logits[k_i] = torch.where(valid_mask, logits[k_i], logits[k_i] - constrain_penalty)
                        else:
                            logits[k_i][~valid_mask] = float("-inf")

                log_probs = F.log_softmax(logits, dim=-1)    # [cur_k, V]

                for k_i, (score, tok_seq) in enumerate(beams):
                    top_probs, top_ids = log_probs[k_i].topk(beam_size)
                    for prob, tok in zip(top_probs.tolist(), top_ids.tolist()):
                        if tok == self.eos_id:
                            done.append((score + prob, tok_seq[1:]))  # strip SOS
                        else:
                            all_candidates.append((score + prob, tok_seq + [tok]))

                if not all_candidates:
                    break

                # Keep top-K
                all_candidates.sort(key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty, reverse=True)
                beams = all_candidates[:beam_size]

            if not done:
                done = [(sc, seq[1:]) for sc, seq in beams]

            best = max(done, key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty)[1]
            results.append(best)

        return results

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check ──────────────────────────────────────────────────
if __name__ == "__main__":
    model = ARModel(vocab_size=120, pretrained=False)
    print(f"Parameters: {model.count_params():,}")

    imgs   = torch.randn(2, 1, 64, 512)
    # labels: SOS + 10 chars + EOS, padded to 14
    labels = torch.zeros(2, 14, dtype=torch.long)
    labels[:, 0] = 1   # SOS
    labels[:, 1:11] = torch.randint(4, 100, (2, 10))
    labels[:, 11] = 2  # EOS
    llens  = torch.tensor([12, 12])

    loss, ce_loss, ctc_loss = model.compute_loss(imgs, labels, llens)
    print(f"AR Total loss: {loss.item():.4f}, CE loss: {ce_loss.item():.4f}, CTC loss: {ctc_loss.item():.4f}")

    preds = model.greedy_decode(imgs, max_len=15)
    print(f"Greedy decode lengths: {[len(p) for p in preds]}")
