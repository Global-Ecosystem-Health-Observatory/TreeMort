import torch
import torchmetrics

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def evaluate(nir_model, test_nir_loader, criterion, device):
    nir_model.eval()

    test_loss = 0.0
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)  # Instantiate PSNR
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # Instantiate SSIM

    with torch.no_grad():
        for rgb_test_batch, nir_test_batch in test_nir_loader:
            rgb_test_batch = rgb_test_batch.to(device)
            nir_test_batch = nir_test_batch.to(device)

            nir_pred = nir_model(rgb_test_batch)

            combined_loss, mse_loss, ssim_loss = criterion(nir_pred, nir_test_batch.unsqueeze(1))
            test_loss += combined_loss.item()

            mae_metric.update(nir_pred, nir_test_batch.unsqueeze(1))
            psnr_metric.update(nir_pred, nir_test_batch.unsqueeze(1))  # Update PSNR metric
            ssim_metric.update(nir_pred, nir_test_batch.unsqueeze(1))  # Update SSIM metric

    test_loss /= len(test_nir_loader)
    avg_mae = mae_metric.compute().item()

    psnr_value = psnr_metric.compute()
    avg_psnr = psnr_value.mean().item() if isinstance(psnr_value, torch.Tensor) else sum(psnr_value) / len(psnr_value)  # Average across channels if tuple

    ssim_value = ssim_metric.compute()
    avg_ssim = ssim_value.mean().item() if isinstance(ssim_value, torch.Tensor) else sum(ssim_value) / len(ssim_value)  # Average across channels if tuple

    mae_metric.reset()
    psnr_metric.reset()
    ssim_metric.reset()

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f} dB")
    print(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")