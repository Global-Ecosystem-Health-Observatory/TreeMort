import os
import torch
import logging

from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0


def train(
    nir_model,
    train_nir_loader,
    val_nir_loader,
    optimizer,
    criterion,
    device,
    num_epochs=10,
    outdir="output",
    patience=5,
):
    early_stopping = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        nir_model.train()

        running_loss = 0.0
        train_loader_tqdm = tqdm(train_nir_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False,)

        for rgb_batch, nir_batch in train_loader_tqdm:
            rgb_batch = rgb_batch.to(device)
            nir_batch = nir_batch.to(device)

            optimizer.zero_grad()

            outputs = nir_model(rgb_batch)

            loss = criterion(outputs, nir_batch.unsqueeze(1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(nir_model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix({"Train Loss": f"{running_loss / len(train_nir_loader):.4f}"})

        nir_model.eval()

        val_loss = 0.0
        val_loader_tqdm = tqdm(val_nir_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False,)
        with torch.no_grad():
            for rgb_val_batch, nir_val_batch in val_loader_tqdm:
                rgb_val_batch = rgb_val_batch.to(device)
                nir_val_batch = nir_val_batch.to(device)

                val_outputs = nir_model(rgb_val_batch)
                val_loss += criterion(val_outputs, nir_val_batch.unsqueeze(1)).item()

        val_loss /= len(val_nir_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(outdir, "best_model.pth")

            torch.save(nir_model.state_dict(), model_path)
            logging.info(f"Best model saved with Validation Loss: {best_val_loss:.4f} at {model_path}")
            
            optimizer_state_path = os.path.join(outdir, "optimizer.pth")
            torch.save(optimizer.state_dict(), optimizer_state_path)
            logging.info(f"Best optimizer saved with Validation Loss: {best_val_loss:.4f} at {optimizer_state_path}")
