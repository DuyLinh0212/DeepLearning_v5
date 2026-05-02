import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


class SliceMHSAEncoder(nn.Module):
    """Transformer-style encoder block over slice tokens."""

    def __init__(self, embed_dim=1280, num_heads=8, dropout=0.1, ff_mult=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, S, D]
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop_attn(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x


class AttentiveSlicePooling(nn.Module):
    """Self-attentive pooling to aggregate slice tokens."""

    def __init__(self, embed_dim=1280):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, S, D]
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)


class EfficientNetB0_SA(nn.Module):
    """EfficientNet-B0 + multi-head self-attention over slice dimension."""

    def __init__(self, num_heads=8, dropout=0.1):
        super().__init__()

        self.axial = _build_efficientnet_b0().features
        self.coronal = _build_efficientnet_b0().features
        self.sagittal = _build_efficientnet_b0().features

        self.pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = 1280

        self.axial_mhsa = SliceMHSAEncoder(feature_dim, num_heads=num_heads, dropout=dropout)
        self.coronal_mhsa = SliceMHSAEncoder(feature_dim, num_heads=num_heads, dropout=dropout)
        self.sagittal_mhsa = SliceMHSAEncoder(feature_dim, num_heads=num_heads, dropout=dropout)

        self.axial_pool = AttentiveSlicePooling(feature_dim)
        self.coronal_pool = AttentiveSlicePooling(feature_dim)
        self.sagittal_pool = AttentiveSlicePooling(feature_dim)

        self.fc = nn.Linear(3 * feature_dim, 1)

    def _extract_slice_features(self, net, x):
        # x can be [S, C, H, W] or [B, S, C, H, W]
        if x.dim() == 4:
            feat = net(x)
            feat = self.pool(feat).view(feat.size(0), -1)  # [S, D]
            return feat.unsqueeze(0)  # [1, S, D]

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = net(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = feat.view(b, s, -1)  # [B, S, D]
        return feat

    def _encode_plane(self, net, mhsa, attn_pool, x):
        feat = self._extract_slice_features(net, x)
        feat = mhsa(feat)
        feat = attn_pool(feat)  # [B, D]
        return feat

    def forward(self, x):
        # Expect list of 3 tensors, each: [B, S, C, H, W] or [S, C, H, W]
        images = x
        axial = self._encode_plane(self.axial, self.axial_mhsa, self.axial_pool, images[0])
        coronal = self._encode_plane(self.coronal, self.coronal_mhsa, self.coronal_pool, images[1])
        sagittal = self._encode_plane(self.sagittal, self.sagittal_mhsa, self.sagittal_pool, images[2])

        feats = torch.cat([axial, coronal, sagittal], dim=1)
        output = self.fc(feats)
        return output
