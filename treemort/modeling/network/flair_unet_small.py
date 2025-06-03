import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class FlairUNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32, crop_size=256):
        super().__init__()
        self.base_ch = base_ch
        self.crop_size = crop_size

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_ch)
        self.enc2 = self.conv_block(base_ch, base_ch * 2)
        self.enc3 = self.conv_block(base_ch * 2, base_ch * 4)

        # Bottleneck
        self.bottleneck = self.conv_block(base_ch * 4, base_ch * 4)

        # Decoder
        self.up2 = self.up_block(base_ch * 4, base_ch * 2)
        self.up1 = self.up_block(base_ch * 2 + base_ch * 4, base_ch)

        # Final
        self.final = nn.Conv2d(base_ch + base_ch * 2, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch)
        )

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            self.conv_block(out_ch, out_ch)
        )

    def forward(self, x):
        features = []

        # Encoder
        enc1 = self.enc1(x)
        features.append(enc1)  # Feature 1

        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        features.append(enc2)  # Feature 2

        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        features.append(enc3)  # Feature 3

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        features.append(bottleneck)  # Feature 4

        # Decoder
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc3], dim=1)
        features.append(dec2)  # Feature 5

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc2], dim=1)
        features.append(dec1)  # Feature 6

        out = self.final(dec1)
        out = F.interpolate(out, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        return out, features