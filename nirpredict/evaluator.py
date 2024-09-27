import torch


def evaluate(nir_model, test_nir_loader, criterion):
    nir_model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for rgb_test_batch, nir_test_batch in test_nir_loader:
            nir_target = nir_model(rgb_test_batch)

            test_loss += criterion(nir_target, nir_test_batch.unsqueeze(1)).item()

    test_loss /= len(test_nir_loader)
    print(f"Test Loss: {test_loss:.4f}")
