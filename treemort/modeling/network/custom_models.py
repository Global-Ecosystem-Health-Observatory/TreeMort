import torch.nn as nn

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
        return super().forward(pixel_values, pixel_mask)
