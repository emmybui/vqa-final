"""
Image Encoder – CNN pretrained (EfficientNet-B4 / ResNet50 / ViT-B/16)
Output: (B, num_regions, feat_dim)  ← grid features cho co-attention

FIX 1: ViT dùng public API thay vì _process_input() (private, có thể break)
FIX 2: GRID_SIZE lấy từ config thay vì hardcode (7, 7)
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
        grid = config.GRID_SIZE  # FIX: không hardcode (7,7)

        if name == "efficientnet_b4":
            base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d((grid, grid))
            return 1792

        elif name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((grid, grid))
            return 2048

        elif name == "vit_b_16":
            base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.backbone = base
            self._vit_mode = True
            self.pool = None
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
        feat = self.backbone(x)
        feat = self.pool(feat)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        feat = self.proj(feat)                            # (B, H*W, out_dim)
        pooled = feat.mean(dim=1)                         # (B, out_dim)
        return feat, pooled

    def _forward_vit(self, x):
        """
        FIX: Dùng public attributes của ViT thay vì _process_input().
        conv_proj, class_token, encoder đều là public và stable.
        """
        b = self.backbone

        # patch projection: (B, hidden_dim, grid_h, grid_w) → (B, num_patches, hidden_dim)
        patches = b.conv_proj(x).flatten(2).transpose(1, 2)

        # prepend class token
        cls = b.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B, 1+num_patches, 768)

        # transformer encoder
        tokens = b.encoder(tokens)                 # (B, 1+num_patches, 768)

        cls_tok   = tokens[:, 0]                   # (B, 768)
        patch_tok = tokens[:, 1:]                  # (B, num_patches, 768)

        feat   = self.proj(patch_tok)              # (B, num_patches, out_dim)
        pooled = self.proj(cls_tok)                # (B, out_dim)
        return feat, pooled


if __name__ == "__main__":
    enc = ImageEncoder(backbone="efficientnet_b4").to(config.DEVICE)
    dummy = torch.randn(2, 3, 224, 224).to(config.DEVICE)
    f, p = enc(dummy)
    print("features:", f.shape, "  pooled:", p.shape)
    # Expected: features (2, 49, 512)  pooled (2, 512)
