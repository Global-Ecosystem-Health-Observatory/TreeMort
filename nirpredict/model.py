import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NIRPredictor(nn.Module):
    def __init__(self):
        super(NIRPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


def build_model(device, outdir="output"):
    nir_model = NIRPredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nir_model.parameters(), lr=0.001)

    load_best_weights(nir_model, optimizer, outdir, device=device)

    return nir_model, criterion, optimizer


def load_best_weights(nir_model, optimizer, outdir="output", device=None):
    model_path = os.path.join(outdir, "best_model.pth")
    if os.path.exists(model_path):
        logger.info(f"Loading best model weights from {model_path}")
        
        map_location = torch.device("cpu") if not torch.cuda.is_available() else device
        nir_model.load_state_dict(torch.load(model_path, map_location=map_location))

        optimizer_state_path = os.path.join(outdir, "optimizer.pth")
        if os.path.exists(optimizer_state_path):
            logger.info(f"Loading best model optimizer state from {optimizer_state_path}")
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=map_location))
            
        return True
    return False