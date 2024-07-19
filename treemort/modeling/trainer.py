import os
import math
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from treemort.utils.callbacks import build_callbacks
from treemort.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from transformers import MaskFormerImageProcessor

preprocessor = MaskFormerImageProcessor(ignore_index=0, do_resize=True, do_rescale=False, do_normalize=False)

def trainer(
    model,
    optimizer,
    criterion,
    metrics,
    train_loader,
    val_loader,
    conf,
    callbacks,
    device,
):
    for epoch in range(conf.epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {}

        train_progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{conf.epochs} [Training]",
            unit="batch",
        )
        for batch_idx, (images, labels) in enumerate(train_progress_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if conf.model == "maskformer":
                outputs = model(images)
        
                target_sizes = [(image.shape[1], image.shape[2]) for image in images]
                predictions = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                predictions = torch.stack([prediction.unsqueeze(0) for prediction in predictions], dim=0).float().to(device)
                
            else:
                predictions = model(images)
            
            loss = criterion(predictions, labels)

            loss.requires_grad = True
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            batch_metrics = metrics(predictions, labels)
            for key, value in batch_metrics.items():
                if key not in train_metrics:
                    train_metrics[key] = 0.0
                train_metrics[key] += value.item()

            train_progress_bar.set_postfix({"Train Loss": train_loss / (batch_idx + 1)})

        # Average train loss and metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_metrics = {}
        val_progress_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{conf.epochs} [Validation]",
            unit="batch",
        )
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_progress_bar):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                if conf.model == "maskformer":
                    target_sizes = [(image.shape[1], image.shape[2]) for image in images]
                    predictions = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                    outputs = torch.stack([prediction.unsqueeze(0) for prediction in predictions], dim=0).float().to(device)
    
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                batch_metrics = metrics(outputs, labels)
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value.item()

                val_progress_bar.set_postfix({"Val Loss": val_loss / (batch_idx + 1)})

        # Average validation loss and metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        print(f"Epoch {epoch + 1}/{conf.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Validation Metrics: {val_metrics}")

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
