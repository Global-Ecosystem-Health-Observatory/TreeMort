import torch.nn as nn
import pytorch_msssim


class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=1)
        self.ssim_weight = ssim_weight

    def forward(self, predicted, target):
        mse = self.mse_loss(predicted, target)
        ssim = 1 - self.ssim_loss(predicted, target)
        combined_loss = mse + self.ssim_weight * ssim
        return combined_loss, mse, ssim