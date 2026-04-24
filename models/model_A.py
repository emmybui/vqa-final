"""
Hướng A:
  VQA_A1 – CNN + BiLSTM + Co-Attention + LSTM Decoder
  VQA_A2 – CNN + BiLSTM + Co-Attention + Transformer Decoder

FIX 1: generate() guard beam_size — LSTMDecoder không nhận beam_size kwarg.
FIX 2: TransformerDecoder nhận full memory sequence (img_feat + txt_feat)
        thay vì 1 context vector, giữ đủ spatial information.
FIX 3: Thêm proj_to_embed để align FUSION_DIM → embed_dim trước khi cat memory.
"""

import torch
import torch.nn as nn

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
        pad_idx: int      = 0,
    ):
        super().__init__()
        self.pad_idx      = pad_idx
        self.decoder_type = decoder_type

        # ── Encoders ──────────────────────────────────────────────────────
        self.img_encoder = ImageEncoder(
            backbone=config.IMG_ENCODER,
            out_dim=config.FUSION_DIM,
        )
        self.txt_encoder = BiLSTMEncoder(
            vocab_size=q_vocab_size,
            out_dim=config.FUSION_DIM,
            pad_idx=pad_idx,
        )

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = CoAttentionFusion(
            dim=config.FUSION_DIM,
            num_heads=config.ATTENTION_HEADS,
        )
        context_dim = config.FUSION_DIM * 2

        # ── Decoder ───────────────────────────────────────────────────────
        if decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                vocab_size  = a_vocab_size,
                context_dim = context_dim,
                pad_idx     = pad_idx,
            )
            # LSTM dùng fused context vector — không cần proj
            self._mem_proj = None

        else:  # "transformer"
            embed_dim = config.FUSION_DIM  # TransformerDecoder embed_dim

            # FIX: Project img_feat và txt_feat về embed_dim nếu cần
            # (hiện FUSION_DIM == embed_dim nên proj là identity-like)
            self._mem_proj = nn.Linear(config.FUSION_DIM, embed_dim) \
                             if config.FUSION_DIM != embed_dim else nn.Identity()

            self.decoder = TransformerAnswerDecoder(
                vocab_size = a_vocab_size,
                embed_dim  = embed_dim,
                pad_idx    = pad_idx,
            )

    # ── Forward (training) ────────────────────────────────────────────────

    def forward(self, image, q_ids, q_len, a_in):
        """
        image : (B, 3, H, W)
        q_ids : (B, T_q)
        q_len : (B, 1)
        a_in  : (B, T_a)  teacher-forcing input
        Returns logits (B, T_a, a_vocab)
        """
        img_feat, _ = self.img_encoder(image)               # (B, Nv, D)
        txt_feat, _ = self.txt_encoder(q_ids, q_len)        # (B, Nq, D)
        txt_pad     = (q_ids == self.pad_idx)               # (B, Nq)

        fused, _, _ = self.fusion(img_feat, txt_feat, txt_pad)  # (B, D*2)

        if self.decoder_type == "lstm":
            logits = self.decoder(fused, a_in)              # (B, T_a, V)

        else:
            # FIX: dùng full sequence (Nv + Nq tokens) làm memory
            # Transformer có thể attend vào từng patch ảnh và từng token text
            memory, mem_mask = self._build_memory(img_feat, txt_feat, txt_pad)
            logits = self.decoder(memory, a_in,
                                  memory_key_padding_mask=mem_mask)

        return logits

    def _build_memory(self, img_feat, txt_feat, txt_pad_mask):
        """
        Ghép img_feat (B, Nv, D) + txt_feat (B, Nq, D) thành memory sequence.
        memory_mask: (B, Nv+Nq) — False ở vị trí img (không bao giờ pad),
                                   True ở vị trí txt pad.
        """
        B, Nv, _ = img_feat.shape
        img_proj = self._mem_proj(img_feat)   # (B, Nv, D)
        txt_proj = self._mem_proj(txt_feat)   # (B, Nq, D)

        memory = torch.cat([img_proj, txt_proj], dim=1)  # (B, Nv+Nq, D)

        # mask: image tokens không bao giờ bị mask
        img_mask = torch.zeros(B, Nv, dtype=torch.bool, device=img_feat.device)
        mem_mask = torch.cat([img_mask, txt_pad_mask], dim=1)  # (B, Nv+Nq)

        return memory, mem_mask

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, image, q_ids, q_len, bos_idx, eos_idx, beam_size: int = 1):
        img_feat, _ = self.img_encoder(image)
        txt_feat, _ = self.txt_encoder(q_ids, q_len)
        txt_pad     = (q_ids == self.pad_idx)
        fused, _, _ = self.fusion(img_feat, txt_feat, txt_pad)

        # FIX: dispatch đúng cho từng loại decoder
        if self.decoder_type == "lstm":
            # LSTMDecoder chỉ greedy, không có beam_size
            return self.decoder.generate(fused, bos_idx, eos_idx)
        else:
            memory, mem_mask = self._build_memory(img_feat, txt_feat, txt_pad)
            return self.decoder.generate(
                memory, bos_idx, eos_idx,
                beam_size=beam_size,
                memory_key_padding_mask=mem_mask,
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Convenient constructors ───────────────────────────────────────────────────

def build_model_A1(q_vocab_size: int, a_vocab_size: int, **kw) -> VQAModel:
    """Hướng A1: LSTM Decoder"""
    return VQAModel(q_vocab_size, a_vocab_size, decoder_type="lstm", **kw)


def build_model_A2(q_vocab_size: int, a_vocab_size: int, **kw) -> VQAModel:
    """Hướng A2: Transformer Decoder"""
    return VQAModel(q_vocab_size, a_vocab_size, decoder_type="transformer", **kw)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 2
    for tag, build_fn in [("A1-LSTM", build_model_A1), ("A2-Transformer", build_model_A2)]:
        m   = build_fn(q_vocab_size=5000, a_vocab_size=3000).to(config.DEVICE)
        img  = torch.randn(B, 3, 224, 224).to(config.DEVICE)
        qid  = torch.randint(1, 5000, (B, config.MAX_Q_LEN)).to(config.DEVICE)
        qlen = torch.randint(5, 20, (B, 1)).to(config.DEVICE)
        ain  = torch.randint(1, 3000, (B, config.MAX_A_LEN)).to(config.DEVICE)
        out  = m(img, qid, qlen, ain)
        print(f"[{tag}] logits: {out.shape}  params: {m.count_parameters():,}")
