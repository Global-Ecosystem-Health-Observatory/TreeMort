import torch
from tqdm import tqdm
import os

def train(nir_model, train_nir_loader, val_nir_loader, optimizer, criterion, device, num_epochs=10, outdir="output"):
    best_val_loss = float("inf")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for epoch in range(num_epochs):
        nir_model.train()

        running_loss = 0.0
        train_loader_tqdm = tqdm(train_nir_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)

        for rgb_batch, nir_batch in train_loader_tqdm:
            rgb_batch = rgb_batch.to(device)
            nir_batch = nir_batch.to(device)

            optimizer.zero_grad()

            outputs = nir_model(rgb_batch)

            loss = criterion(outputs, nir_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix({"Train Loss": f"{running_loss / len(train_nir_loader):.4f}"})

        nir_model.eval()

        val_loss = 0.0
        val_loader_tqdm = tqdm(val_nir_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)
        with torch.no_grad():
            for rgb_val_batch, nir_val_batch in val_loader_tqdm:
                rgb_val_batch = rgb_val_batch.to(device)
                nir_val_batch = nir_val_batch.to(device)

                val_outputs = nir_model(rgb_val_batch)
                val_loss += criterion(val_outputs, nir_val_batch.unsqueeze(1)).item()

        val_loss /= len(val_nir_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(outdir, "best_model.pth")
            torch.save(nir_model.state_dict(), model_path)
            print(f"Best model saved with Validation Loss: {best_val_loss:.4f} at {model_path}")