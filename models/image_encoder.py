"""
Image Encoder – CNN pretrained (EfficientNet-B4 / ResNet-50 / ViT-B/16)
Output: (B, num_regions, feat_dim)  ← grid features cho co-attention
"""

import torch
import torch.nn as nn
from torchvision import models
import config


class ImageEncoder(nn.Module):
    """
    Trả về:
        features : (B, num_regions, feat_dim)   — dùng cho co-attention
        pooled   : (B, feat_dim)                — global feature
    """

    def __init__(
        self,
        backbone: str = config.IMG_ENCODER,
        out_dim: int = config.FUSION_DIM,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        feat_dim = self._build_backbone(backbone)

        # projection → FUSION_DIM
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
        )
        self.out_dim = out_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ── builders ────────────────────────────────────────────────────────────

    def _build_backbone(self, name: str) -> int:
        """Xây backbone, trả về feat_dim của spatial feature."""
        if name == "efficientnet_b4":
            base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            # lấy features trước avgpool
            self.backbone = base.features          # (B, 1792, 7, 7) với 224x224
            self.pool     = nn.AdaptiveAvgPool2d((7, 7))
            return 1792

        elif name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-2])  # remove avgpool + fc
            self.pool     = nn.AdaptiveAvgPool2d((7, 7))
            return 2048

        elif name == "vit_b_16":
            # ViT: dùng patch tokens  (B, 196, 768)
            base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.backbone = base
            self._vit_mode = True
            self.pool      = None
            return 768

        else:
            raise ValueError(f"Unknown backbone: {name}")

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, H, W)
        returns:
            features : (B, num_regions, out_dim)
            pooled   : (B, out_dim)
        """
        if self.backbone_name == "vit_b_16":
            return self._forward_vit(x)
        return self._forward_cnn(x)

    def _forward_cnn(self, x):
        feat = self.backbone(x)               # (B, C, h, w)
        feat = self.pool(feat)                 # (B, C, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, C)
        feat = self.proj(feat)                # (B, 49, out_dim)
        pooled = feat.mean(dim=1)             # (B, out_dim)
        return feat, pooled

    def _forward_vit(self, x):
        # ViT internal: get patch embeddings
        enc = self.backbone._process_input(x)                  # (B, 196, 768)
        batch_class_token = self.backbone.class_token.expand(x.shape[0], -1, -1)
        enc = torch.cat([batch_class_token, enc], dim=1)        # (B, 197, 768)
        enc = self.backbone.encoder(enc)                        # (B, 197, 768)
        cls_tok = enc[:, 0]                                     # (B, 768)
        patch_tok = enc[:, 1:]                                  # (B, 196, 768)
        feat   = self.proj(patch_tok)                           # (B, 196, out_dim)
        pooled = self.proj(cls_tok)                             # (B, out_dim)
        return feat, pooled


if __name__ == "__main__":
    enc = ImageEncoder(backbone="efficientnet_b4").to(config.DEVICE)
    dummy = torch.randn(2, 3, 224, 224).to(config.DEVICE)
    f, p = enc(dummy)
    print("features:", f.shape, "  pooled:", p.shape)
    # Expected: features (2, 49, 512)  pooled (2, 512)
