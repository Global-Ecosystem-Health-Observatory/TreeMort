import os
import math
import torch
from torch.utils.data import DataLoader

from treeseg.utils.callbacks import build_callbacks
from treeseg.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def trainer(
    model, optimizer, criterion, metrics, train_loader, val_loader, conf, callbacks
):

    for epoch in range(conf.epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {}
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.float()  # Ensure labels are float
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_metrics = metrics(outputs, labels)
            for key, value in batch_metrics.items():
                if key not in train_metrics:
                    train_metrics[key] = 0.0
                train_metrics[key] += value.item()

        # Average train loss and metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_metrics = {}
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.float()  # Ensure labels are float
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                batch_metrics = metrics(outputs, labels)
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value.item()

        # Average validation loss and metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        # Print epoch metrics
        print(f"Epoch {epoch + 1}/{conf.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Validation Metrics: {val_metrics}")

        # Call callbacks
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback(epoch + 1, model, optimizer, val_loss)
            elif isinstance(callback, ReduceLROnPlateau):
                callback(val_loss)
            elif isinstance(callback, EarlyStopping):
                callback(epoch + 1, val_loss)
                if callback.stop_training:
                    print("Early stopping")
                    return
