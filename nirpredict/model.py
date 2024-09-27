import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_model(device):
    nir_model = NIRPredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nir_model.parameters(), lr=0.001)

    return nir_model, criterion, optimizer
