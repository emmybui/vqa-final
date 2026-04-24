"""
Fusion Module – Co-Attention (Image ↔ Text)
Lấy cảm hứng từ: "Hierarchical Question-Image Co-Attention for VQA" (Lu et al. 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CoAttentionFusion(nn.Module):
    """
    Co-Attention giữa image regions (V) và text tokens (Q).
    Cho mỗi chiều: tính attention trọng số của bên kia, rồi pool.

    Input:
        img_feat : (B, Nv, D)  – image region features
        txt_feat : (B, Nq, D)  – text token features
    Output:
        fused    : (B, D*2)    – concat attended image + attended text
    """

    def __init__(self, dim: int = config.FUSION_DIM, num_heads: int = config.ATTENTION_HEADS):
        super().__init__()
        self.dim = dim

        # Image-guided text attention
        self.q_attn = nn.MultiheadAttention(dim, num_heads, dropout=config.DROPOUT, batch_first=True)
        # Text-guided image attention
        self.v_attn = nn.MultiheadAttention(dim, num_heads, dropout=config.DROPOUT, batch_first=True)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        # joint projection
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
        txt_key_pad_mask: torch.Tensor = None,   # (B, Nq) True=pad
    ):
        """
        Returns:
            fused   : (B, dim*2)
            v_att_w : (B, Nv)  image attention weights (viz)
            q_att_w : (B, Nq)  text  attention weights (viz)
        """
        # Text attending to Image  (query=text, key/value=image)
        q_ctx, q_w = self.q_attn(
            query=txt_feat,
            key=img_feat,
            value=img_feat,
        )                                             # q_ctx: (B, Nq, D)
        q_ctx = self.norm_q(txt_feat + q_ctx)
        q_pool = q_ctx.mean(dim=1)                    # (B, D) mean-pool over tokens

        # Image attending to Text  (query=image, key/value=text)
        v_ctx, v_w = self.v_attn(
            query=img_feat,
            key=txt_feat,
            value=txt_feat,
            key_padding_mask=txt_key_pad_mask,
        )                                             # v_ctx: (B, Nv, D)
        v_ctx = self.norm_v(img_feat + v_ctx)
        v_pool = v_ctx.mean(dim=1)                    # (B, D)

        fused = self.out_proj(torch.cat([v_pool, q_pool], dim=-1))  # (B, D*2)

        # collapse attention weights to per-token (mean over heads)
        # q_w: (B, Nq, Nv) → (B, Nv)
        v_att_w = q_w.mean(dim=1)    # image importance seen by text
        q_att_w = v_w.mean(dim=1)    # text importance seen by image

        return fused, v_att_w, q_att_w


class SimpleConcatFusion(nn.Module):
    """Fusion đơn giản: concat pooled image + pooled text."""

    def __init__(self, dim: int = config.FUSION_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, img_pooled, txt_pooled):
        return self.proj(torch.cat([img_pooled, txt_pooled], dim=-1))


if __name__ == "__main__":
    B, Nv, Nq, D = 4, 49, 32, config.FUSION_DIM
    img = torch.randn(B, Nv, D)
    txt = torch.randn(B, Nq, D)
    co = CoAttentionFusion()
    fused, vw, qw = co(img, txt)
    print("fused:", fused.shape)      # (4, 1024)
    print("v_att:", vw.shape)         # (4, 49)
    print("q_att:", qw.shape)         # (4, 32)
