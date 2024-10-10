import torch

def predict(nir_model, rgb_input, device):
    nir_model.eval()

    rgb_input = rgb_input.to(device)

    with torch.no_grad():
        logits = nir_model(rgb_input)

        nir_pred = torch.clamp(torch.relu(logits), 0, 1)

    return nir_pred