import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_val=1.0):
        return GradientReversalFunction.apply(x, lambda_val)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, features, lambda_val=1.0):
        x = self.grl(features, lambda_val)
        x = self.pool(x)
        return self.classifier(x)  # [B, 1] logits


class FlairUNetDANN(nn.Module):
    """Wraps CombinedModel with a domain discriminator head for DANN training."""

    def __init__(self, base_model, bottleneck_channels=512):
        super().__init__()
        self.base = base_model
        self.discriminator = DomainDiscriminator(in_channels=bottleneck_channels)

    def forward(self, x):
        return self.base(x)  # (logits, encoder_features) — same interface as CombinedModel

    def discriminate(self, encoder_features, lambda_val=1.0):
        bottleneck = encoder_features[-1]  # deepest encoder level: [B, 512, H', W']
        return self.discriminator(bottleneck, lambda_val)
