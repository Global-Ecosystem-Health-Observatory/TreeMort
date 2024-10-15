import torch

from tqdm import tqdm

from treemort.training.output_processing import process_model_output
from treemort.utils.loss import compute_weighted_domain_loss


def train_one_epoch(model, optimizer, seg_criterion, metrics, train_loader, conf, device, num_finnish, num_us):
    model.train()
    train_loss = 0.0
    train_metrics = {}
    train_seg_loss = 0.0
    train_domain_loss = 0.0

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)
    
    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, seg_labels, domain_labels) in enumerate(train_progress_bar):
        images = images.to(device)
        seg_labels = seg_labels.to(device)
        domain_labels = domain_labels.to(device)

        optimizer.zero_grad()

        seg_output, domain_output = model(images)

        seg_loss = seg_criterion(seg_output, seg_labels, class_weights=class_weights)

        domain_loss = compute_weighted_domain_loss(domain_output, domain_labels, num_finnish, num_us)

        total_loss = seg_loss + conf.lambda_adv * domain_loss

        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_seg_loss += seg_loss.item()
        train_domain_loss += domain_loss.item()

        pred_probs = torch.sigmoid(seg_output)
        batch_metrics = metrics(pred_probs, seg_labels)
        for key, value in batch_metrics.items():
            if key not in train_metrics:
                train_metrics[key] = 0.0
            train_metrics[key] += value.item()

        train_progress_bar.set_postfix({
            "Total Loss": train_loss / (batch_idx + 1),
            "Seg Loss": train_seg_loss / (batch_idx + 1),
            "Domain Loss": train_domain_loss / (batch_idx + 1),
        })

    train_loss /= len(train_loader)
    train_seg_loss /= len(train_loader)
    train_domain_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, train_metrics