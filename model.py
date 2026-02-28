import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121


# =========================
# Squash (Capsule activation)
# =========================
class Squash(nn.Module):
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * (x / (norm + 1e-8))


# =========================
# CBAM
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# =========================
# DenseNet + CBAM + Capsule
# =========================
class DenseNet_CBAM_Capsule(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Backbone EXACTEMENT comme training
        self.backbone = densenet121(weights=None)
        self.features = self.backbone.features

        # CBAM sur 1024 canaux
        self.cbam = CBAM(in_channels=1024)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Capsule part
        self.project = nn.Linear(1024, 512)
        self.capsules = nn.Linear(512, 16 * 16)
        self.squash = Squash()

        self.classifier = nn.Linear(16 * 16, num_classes)

    def forward(self, x):

        # DenseNet features
        x = self.features(x)
        x = x = F.relu(x)

        # CBAM
        x = self.cbam(x)

        # Global pooling
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # Capsule projection
        x = self.project(x)
        x = self.capsules(x)

        x = x.view(-1, 16, 16)
        x = self.squash(x)
        x = x.view(x.size(0), -1)

        return self.classifier(x)