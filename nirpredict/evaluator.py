import torch
import torchmetrics

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


def evaluate(nir_model, test_nir_loader, criterion, device):
    nir_model.eval()

    test_loss = 0.0
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for rgb_test_batch, nir_test_batch in test_nir_loader:
            rgb_test_batch = rgb_test_batch.to(device)
            nir_test_batch = nir_test_batch.to(device)

            nir_pred = nir_model(rgb_test_batch)

            test_loss += criterion(nir_pred, nir_test_batch.unsqueeze(1)).item()

            mae_metric.update(nir_pred, nir_test_batch.unsqueeze(1))

            total_psnr += peak_signal_noise_ratio(nir_pred, nir_test_batch.unsqueeze(1))

            total_ssim += structural_similarity_index_measure(nir_pred, nir_test_batch.unsqueeze(1))

    test_loss /= len(test_nir_loader)
    
    avg_mae = mae_metric.compute().item()
    avg_psnr = total_psnr / len(test_nir_loader)
    avg_ssim = total_ssim / len(test_nir_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f} dB")
    print(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")