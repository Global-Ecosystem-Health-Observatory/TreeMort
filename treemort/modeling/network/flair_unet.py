import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from huggingface_hub import hf_hub_download

from treemort.modeling.network.unet_mtd import smp_unet_mtd


class PretrainedUNetModel:
    def __init__(
        self,
        repo_id,
        filename,
        architecture="unet",
        encoder="resnet34",
        n_channels=4,
        n_classes=15,
        use_metadata=False,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.architecture = architecture
        self.encoder = encoder
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_metadata = use_metadata

        self.checkpoint_path = hf_hub_download(
            repo_id=self.repo_id, filename=self.filename
        )

        self.model = self._initialize_model()

        self._load_pretrained_weights()

    def _initialize_model(self):
        model = smp_unet_mtd(
            architecture=self.architecture,
            encoder=self.encoder,
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            use_metadata=self.use_metadata,
        )
        return model

    def _load_pretrained_weights(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path), strict=False)

    def get_model(self):
        return self.model


class SelfAttentionUNetDecoder(nn.Module):
    def __init__(
        self,
        n_classes=1,
        depth=5,  # Ensure depth is sufficient for the desired output size
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode="upconv",
        kernel_size=3,
    ):
        super(SelfAttentionUNetDecoder, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth

        prev_channels = 2 ** (wf + depth - 1)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        # Final layer to convert to desired output channels (n_classes)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, encoder_features):
        for i, up in enumerate(self.up_path):
            x = up(x, encoder_features[-i - 2])
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

    def forward(self, x, bridge):
        up = self.up(x)
        # Handle padding to ensure dimensions match correctly
        diffY = bridge.size()[2] - up.size()[2]
        diffX = bridge.size()[3] - up.size()[3]
        up = F.pad(up, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class FeatureExtractor(nn.Module):
    def __init__(self, model, use_metadata=False):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.use_metadata = use_metadata
        self.features = []

        # Hook the layers of the encoder
        layers = [
            self.model.seg_model.encoder.layer1[-1],
            self.model.seg_model.encoder.layer2[-1],
            self.model.seg_model.encoder.layer3[-1],
            self.model.seg_model.encoder.layer4[-1],
        ]
        for layer in layers:
            layer.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features.append(output)

    def forward(self, x, met=None):
        self.features = []
        if self.use_metadata:
            self.model(x, met)
        else:
            self.model(x)
        return self.features


class CombinedModel(nn.Module):
    def __init__(self, pretrained_model, n_classes=3, output_size=256):
        super(CombinedModel, self).__init__()
        self.feature_extractor = FeatureExtractor(pretrained_model)
        self.decoder = SelfAttentionUNetDecoder(
            n_classes=n_classes,
            depth=4,
            wf=6,
            padding=True,
            batch_norm=False,
            up_mode="upconv",
            kernel_size=3,
        )
        # Additional upsampling layer
        self.upsample = nn.Upsample(
            size=(output_size, output_size), mode="bilinear", align_corners=False
        )

    def forward(self, x):
        encoder_features = self.feature_extractor(x)
        decoder_output = self.decoder(encoder_features[-1], encoder_features)
        upsampled_output = self.upsample(decoder_output)
        return upsampled_output
