import torch
from torch import nn
import torch.nn.functional as F

class MultiScaleAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels=9,
        n_classes=3,
        depth=3,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode="upconv",
        kernel_size=3,
    ):
        super(MultiScaleAttentionUNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        # Downsampling path
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(
                    prev_channels, 2 ** (wf + i), padding, batch_norm, kernel_size
                )
            )
            prev_channels = 2 ** (wf + i)

        # Upsampling path
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, kernel_size=3):
        super(SelfAttentionBlock, self).__init__()

        padding_ = int(padding) * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_size,
            out_size,
            kernel_size=kernel_size,
            padding=padding_,
            padding_mode="reflect",
        )
        self.attention = nn.Conv2d(
            in_size,
            out_size,
            kernel_size=kernel_size,
            padding=padding_,
            padding_mode="reflect",
            bias=False,
        )
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, kernel_size=3):
        super(UNetConvBlock, self).__init__()

        self.self_attention1 = SelfAttentionBlock(
            in_size, out_size, padding, kernel_size
        )
        self.self_attention2 = SelfAttentionBlock(
            out_size, out_size, padding, kernel_size
        )
        self.batch_norm = batch_norm

        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(out_size, out_size, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(out_size, out_size, kernel_size=5, padding=2)
        
        # Final convolution to reduce channels
        self.final_conv = nn.Conv2d(3 * out_size, out_size, kernel_size=1, padding=0)

        if batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(out_size)
            self.batch_norm2 = nn.BatchNorm2d(out_size)
            self.batch_norm_conv1x1 = nn.BatchNorm2d(out_size)
            self.batch_norm_conv3x3 = nn.BatchNorm2d(out_size)
            self.batch_norm_conv5x5 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        x1 = F.relu(self.self_attention1(x))
        if self.batch_norm:
            x1 = self.batch_norm1(x1)

        x2 = F.relu(self.self_attention2(x1))
        if self.batch_norm:
            x2 = self.batch_norm2(x2)

        # Apply multi-scale convolutions
        x1x1 = self.conv1x1(x2)
        x3x3 = self.conv3x3(x2)
        x5x5 = self.conv5x5(x2)

        if self.batch_norm:
            x1x1 = self.batch_norm_conv1x1(x1x1)
            x3x3 = self.batch_norm_conv3x3(x3x3)
            x5x5 = self.batch_norm_conv5x5(x5x5)

        # Concatenate multi-scale features and reduce channels
        x_out = torch.cat([x1x1, x3x3, x5x5], dim=1)
        x_out = self.final_conv(x_out)
        return x_out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
