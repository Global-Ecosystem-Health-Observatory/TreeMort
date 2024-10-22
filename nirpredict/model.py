import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nirpredict.loss import CombinedLoss
from treemort.utils.logger import get_logger

logger = get_logger(__name__)


class NIRPredictor(nn.Module):
    def __init__(self):
        super(NIRPredictor, self).__init__()

        # Encoder part - Downsampling
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Added Layer
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(2, 2)  # Reducing the size by half

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(2, 2)  # Reducing the size by half again

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),  # Increase to 512 filters
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Decoder part - Upsampling
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Input is now 512, output is 256

        self.decoder1 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),  # 256 (from upsample1) + 128 (from encoder2)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 (from upsample2) + 64 (from encoder1)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Final layer to output the NIR band
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        x_pool1 = self.pool1(x1)

        x2 = self.encoder2(x_pool1)
        x_pool2 = self.pool2(x2)

        # Bottleneck
        x_bottleneck = self.bottleneck(x_pool2)

        # Decoder path with skip connections
        x_upsample1 = self.upsample1(x_bottleneck)
        x_concat1 = torch.cat([x_upsample1, x2], dim=1)  # Skip connection
        x3 = self.decoder1(x_concat1)

        x_upsample2 = self.upsample2(x3)
        x_concat2 = torch.cat([x_upsample2, x1], dim=1)  # Skip connection
        x4 = self.decoder2(x_concat2)

        # Final output layer
        output = self.final_conv(x4)

        return output
        

def build_model(device, outdir="output"):
    nir_model = NIRPredictor().to(device)

    criterion = CombinedLoss(ssim_weight=0.1)
    optimizer = torch.optim.Adam(nir_model.parameters(), lr=0.005)

    load_best_weights(nir_model, optimizer, outdir, device=device)

    return nir_model, criterion, optimizer


def load_best_weights(nir_model, optimizer, outdir="output", device=None):
    model_path = os.path.join(outdir, "best_model.pth")
    if os.path.exists(model_path):
        logger.info(f"Loading best model weights from {model_path}")

        map_location = torch.device("cpu") if not torch.cuda.is_available() else device
        nir_model.load_state_dict(torch.load(model_path, map_location=map_location, weights_only=True))

        optimizer_state_path = os.path.join(outdir, "optimizer.pth")
        if os.path.exists(optimizer_state_path):
            logger.info(f"Loading best model optimizer state from {optimizer_state_path}")
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=map_location, weights_only=True))

        return True
    return False
