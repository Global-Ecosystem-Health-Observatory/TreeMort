# Adapted from https://github.com/jvanvugt/pytorch-unet replacing the convolutions in the downwards path with
# self attention layers

import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionUNet(nn.Module):
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
        """
        Modified version of the U-Net to also use selfattention in its layers
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597


        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(SelfAttentionUNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(
                    prev_channels, 2 ** (wf + i), padding, batch_norm, kernel_size
                )
            )
            prev_channels = 2 ** (wf + i)

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
            print('A0', i, x.shape)
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
            print('A1', i, x.shape)

        for i, up in enumerate(self.up_path):
            print('B0', i, x.shape)
            x = up(x, blocks[-i - 1])
            print('B1', i, x.shape)

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
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(out_size)
            self.batch_norm2 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        x = F.relu(self.self_attention1(x))
        if self.batch_norm:
            x = self.batch_norm1(x)

        x = F.relu(self.self_attention2(x))
        if self.batch_norm:
            x = self.batch_norm2(x)
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
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
