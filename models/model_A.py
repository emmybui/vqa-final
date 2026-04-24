"""
Hướng A:
  VQA_A1 – CNN + BiLSTM + Co-Attention + LSTM Decoder
  VQA_A2 – CNN + BiLSTM + Co-Attention + Transformer Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.image_encoder import ImageEncoder
from models.text_encoder  import BiLSTMEncoder
from models.fusion        import CoAttentionFusion
from models.decoder       import LSTMDecoder, TransformerAnswerDecoder


class VQAModel(nn.Module):
    """
    Base class cho cả A1 và A2.
    decoder_type: "lstm" | "transformer"
    """

    def __init__(
        self,
        q_vocab_size: int,
        a_vocab_size: int,
        decoder_type: str = "lstm",
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx      = pad_idx
        self.decoder_type = decoder_type

        # ── Encoders ──
        self.img_encoder = ImageEncoder(
            backbone=config.IMG_ENCODER,
            out_dim=config.FUSION_DIM,
        )
        self.txt_encoder = BiLSTMEncoder(
            vocab_size=q_vocab_size,
            out_dim=config.FUSION_DIM,
            pad_idx=pad_idx,
        )

        # ── Fusion ──
        self.fusion = CoAttentionFusion(
            dim=config.FUSION_DIM,
            num_heads=config.ATTENTION_HEADS,
        )
        # fused dim = FUSION_DIM * 2

        # ── Decoder ──
        context_dim = config.FUSION_DIM * 2
        if decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                vocab_size=a_vocab_size,
                context_dim=context_dim,
                pad_idx=pad_idx,
            )
        else:
            self.decoder = TransformerAnswerDecoder(
                vocab_size=a_vocab_size,
                context_dim=context_dim,
                pad_idx=pad_idx,
            )

    def forward(self, image, q_ids, q_len, a_in):
        """
        image : (B, 3, H, W)
        q_ids : (B, T_q)
        q_len : (B, 1)
        a_in  : (B, T_a)  teacher-forcing input
        Returns logits (B, T_a, a_vocab)
        """
        # encode
        img_feat, _      = self.img_encoder(image)            # (B, Nv, D)
        txt_feat, _      = self.txt_encoder(q_ids, q_len)     # (B, Nq, D)

        # pad mask for text
        txt_pad_mask = (q_ids == self.pad_idx)                # (B, Nq)

        # fuse
        fused, _, _      = self.fusion(img_feat, txt_feat, txt_pad_mask)  # (B, D*2)

        # decode
        logits = self.decoder(fused, a_in)                    # (B, T_a, V)
        return logits

    @torch.no_grad()
    def generate(self, image, q_ids, q_len, bos_idx, eos_idx, beam_size=1):
        img_feat, _ = self.img_encoder(image)
        txt_feat, _ = self.txt_encoder(q_ids, q_len)
        txt_pad     = (q_ids == self.pad_idx)
        fused, _, _ = self.fusion(img_feat, txt_feat, txt_pad)
        return self.decoder.generate(fused, bos_idx, eos_idx, beam_size=beam_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── convenient constructors ────────────────────────────────────────────────

def build_model_A1(q_vocab_size: int, a_vocab_size: int, **kw) -> VQAModel:
    """Hướng A1: LSTM Decoder"""
    return VQAModel(q_vocab_size, a_vocab_size, decoder_type="lstm", **kw)


def build_model_A2(q_vocab_size: int, a_vocab_size: int, **kw) -> VQAModel:
    """Hướng A2: Transformer Decoder"""
    return VQAModel(q_vocab_size, a_vocab_size, decoder_type="transformer", **kw)


# ─── quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 2
    for tag, build_fn in [("A1-LSTM", build_model_A1), ("A2-Transformer", build_model_A2)]:
        m = build_fn(q_vocab_size=5000, a_vocab_size=3000).to(config.DEVICE)
        img  = torch.randn(B, 3, 224, 224).to(config.DEVICE)
        qid  = torch.randint(0, 5000, (B, config.MAX_Q_LEN)).to(config.DEVICE)
        qlen = torch.randint(5, 20, (B, 1)).to(config.DEVICE)
        ain  = torch.randint(0, 3000, (B, config.MAX_A_LEN)).to(config.DEVICE)
        out  = m(img, qid, qlen, ain)
        print(f"[{tag}] logits: {out.shape}  params: {m.count_parameters():,}")
