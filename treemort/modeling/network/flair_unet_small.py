import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MobileNetV2Encoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        self.initial = mobilenet.features[0]

        # Modify the first conv layer to accept 4 channels
        orig_conv = self.initial[0]
        new_conv = nn.Conv2d(
            input_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_conv.weight
            new_conv.weight[:, 3] = orig_conv.weight[:, 0]
        self.initial[0] = new_conv

        # Use selected layers from mobilenet
        self.encoder_layers = nn.Sequential(*mobilenet.features[1:])

    def forward(self, x):
        x = self.initial(x)
        features = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return features


class CompressedUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.encoder = MobileNetV2Encoder(input_channels)

        # Corrected input channels for each Up block based on encoder features:
        # up1: 1280 (bottleneck) + 320 (skip) = 1600
        # up2: 320 + 160 = 480
        # up3: 160 + 96 = 256
        # up4: 32 + 16 = 48 (using feats[1] which is 16 channels)
        self.up1 = Up(1600, 320)
        self.up2 = Up(480, 96)
        self.up3 = Up(256, 32)
        self.up4 = Up(48, 24)
        self.outc = OutConv(24, output_channels)

    def forward(self, x):
        x_input = x
        feats = self.encoder(x)
        # feats: [0]=24, [1]=32, [2]=32, [3]=96, [4]=320, [5]=1280 (if 6 total)
        # Map: up1(feats[-1], feats[-2]) -> [1280, 320], up2(x, feats[-3]) -> [320, 160], etc.
        x = self.up1(feats[-1], feats[-2])      # [1280 + 320]
        x = self.up2(x, feats[-3])              # [320 + 160]
        x = self.up3(x, feats[-4])              # [160 + 96]
        x = self.up4(x, feats[1])               # [32 + 32], using feats[1] (32 channels)
        x = self.outc(x)
        x = F.interpolate(x, size=(x_input.shape[2], x_input.shape[3]), mode='bilinear', align_corners=False)
        return x, feats