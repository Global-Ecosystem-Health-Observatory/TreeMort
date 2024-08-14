import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    MaskFormerModel,
    MaskFormerForInstanceSegmentation,
    DetrModel,
    DetrForSegmentation,
    BeitModel,
    BeitForSemanticSegmentation,
)


class CustomMaskFormer(MaskFormerForInstanceSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.model = MaskFormerModel(config)
        self.conv1 = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, pixel_values, pixel_mask=None):
        # Map the 4-channel input to 3 channels
        pixel_values = self.conv1(pixel_values)
        return super().forward(pixel_values, pixel_mask)


class CustomDetr(DetrForSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.model = DetrModel(config)
        self.conv1 = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, pixel_values, pixel_mask=None):
        # Map the 4-channel input to 3 channels
        pixel_values = self.conv1(pixel_values)
        return super().forward(pixel_values, pixel_mask)


class CustomBeit(BeitForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.model = BeitModel(config)
        self.conv1 = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, pixel_values, pixel_mask=None):
        # Map the 4-channel input to 3 channels
        pixel_values = self.conv1(pixel_values)
        
        outputs = super().forward(pixel_values, pixel_mask)

        # Upsample the logits (160x160) to match the target size (e.g., 640x640)
        upsampled_logits = F.interpolate(outputs.logits, size=(pixel_values.shape[2], pixel_values.shape[3]), mode='bilinear', align_corners=False)

        outputs.logits = upsampled_logits
        return outputs