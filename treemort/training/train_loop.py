import torch

from tqdm import tqdm

from treemort.training.output_processing import process_model_output
from treemort.utils.loss import ewc_loss


def train_one_epoch(model, optimizer, criterion, metrics, train_loader, fisher_information, optimal_parameters, lambda_ewc, conf, device):
    model.train()
    train_loss = 0.0
    train_metrics = {}

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)
    
    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = process_model_output(model, images, conf)
        
        loss = criterion(logits, labels, class_weights=class_weights)

        loss.backward()
        optimizer.step()

        ewc_penalty = ewc_loss(model, fisher_information, optimal_parameters, lambda_ewc)
        total_loss = loss.item() + ewc_penalty

        train_loss += total_loss

        pred_probs = torch.sigmoid(logits)

        batch_metrics = metrics(pred_probs, labels)
        for key, value in batch_metrics.items():
            if key not in train_metrics:
                train_metrics[key] = 0.0
            train_metrics[key] += value.item()

        train_progress_bar.set_postfix({"Train Loss": train_loss / (batch_idx + 1)})

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, train_metrics
