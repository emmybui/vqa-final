import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CoAttentionFusion(nn.Module):
    def __init__(self, dim: int = config.FUSION_DIM, num_heads: int = config.ATTENTION_HEADS):
        super().__init__()
        self.dim = dim

        # Text-query → attend image regions
        self.q_attn = nn.MultiheadAttention(dim, num_heads, dropout=config.DROPOUT, batch_first=True)
        # Image-query → attend text tokens
        self.v_attn = nn.MultiheadAttention(dim, num_heads, dropout=config.DROPOUT, batch_first=True)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

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
        txt_key_pad_mask: torch.Tensor = None,   # (B, Nq) True = pad
    ):
        # query = txt, key/value = img → mỗi token text chú ý vùng ảnh nào
        q_ctx, q_w = self.q_attn(
            query=txt_feat,
            key=img_feat,
            value=img_feat,
        )                                              # q_ctx: (B, Nq, D)
                                                       # q_w  : (B, Nq, Nv)
        q_ctx  = self.norm_q(txt_feat + q_ctx)
        q_pool = q_ctx.mean(dim=1)                     # (B, D) mean-pool qua tokens

        # query = img, key/value = txt → mỗi vùng ảnh chú ý token nào
        v_ctx, v_w = self.v_attn(
            query=img_feat,
            key=txt_feat,
            value=txt_feat,
            key_padding_mask=txt_key_pad_mask,
        )                                              # v_ctx: (B, Nv, D)
                                                       # v_w  : (B, Nv, Nq)
        v_ctx  = self.norm_v(img_feat + v_ctx)
        v_pool = v_ctx.mean(dim=1)                     # (B, D)

        fused = self.out_proj(torch.cat([v_pool, q_pool], dim=-1))  # (B, D*2)

        img_attn_w = q_w.mean(dim=1)   # (B, Nv) – image region importance
        txt_attn_w = v_w.mean(dim=1)   # (B, Nq) – text token importance

        return fused, img_attn_w, txt_attn_w


class SimpleConcatFusion(nn.Module):

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
    co  = CoAttentionFusion()
    fused, img_attn_w, txt_attn_w = co(img, txt)
    print("fused      :", fused.shape)        # (4, 1024)
    print("img_attn_w :", img_attn_w.shape)   # (4, 49)
    print("txt_attn_w :", txt_attn_w.shape)   # (4, 32)
