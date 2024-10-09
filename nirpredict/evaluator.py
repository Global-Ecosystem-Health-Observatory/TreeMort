import torch
import torchmetrics

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(nir_model, test_nir_loader, criterion, device):
    nir_model.eval()

    test_loss = 0.0
    total_mse_loss = 0.0
    total_ssim_loss = 0.0
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for rgb_test_batch, nir_test_batch in test_nir_loader:
            rgb_test_batch = rgb_test_batch.to(device)
            nir_test_batch = nir_test_batch.to(device)

            nir_pred = nir_model(rgb_test_batch)

            combined_loss, mse_loss, ssim_loss = criterion(nir_pred, nir_test_batch.unsqueeze(1))
            test_loss += combined_loss.item()
            total_mse_loss += mse_loss.item()
            total_ssim_loss += ssim_loss.item()

            mae_metric.update(nir_pred, nir_test_batch.unsqueeze(1))

            total_psnr += peak_signal_noise_ratio(nir_pred, nir_test_batch.unsqueeze(1)).item()
            total_ssim += structural_similarity_index_measure(nir_pred, nir_test_batch.unsqueeze(1)).item()

    test_loss /= len(test_nir_loader)
    avg_mse_loss = total_mse_loss / len(test_nir_loader)
    avg_ssim_loss = total_ssim_loss / len(test_nir_loader)
    avg_mae = mae_metric.compute().item()
    avg_psnr = total_psnr / len(test_nir_loader)
    avg_ssim = total_ssim / len(test_nir_loader)

    logger.info(f"Test Loss (Combined): {test_loss:.4f}")
    logger.info(f"MSE Loss Component: {avg_mse_loss:.4f}")
    logger.info(f"SSIM Loss Component: {avg_ssim_loss:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
    logger.info(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f} dB")
    logger.info(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")