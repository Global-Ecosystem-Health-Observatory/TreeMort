import torch

from tqdm import tqdm

from treemort.training.output_processing import process_model_output


def validate_one_epoch(model, seg_criterion, domain_criterion, metrics, val_loader, conf, device):
    model.eval()
    val_loss = 0.0
    val_metrics = {}

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)

    val_progress_bar = tqdm(val_loader, desc=f"Validation", unit="batch")
    
    with torch.no_grad():
        for batch_idx, (images, seg_labels, domain_labels) in enumerate(val_progress_bar):
            images = images.to(device)
            seg_labels = seg_labels.to(device)
            domain_labels = domain_labels.to(device)

            seg_logits, domain_logits = model(images)

            seg_loss = seg_criterion(seg_logits, seg_labels, class_weights=class_weights)

            domain_loss = domain_criterion(domain_logits, domain_labels)

            total_loss = seg_loss + conf.lambda_adv * domain_loss

            val_loss += total_loss.item()

            pred_probs = torch.sigmoid(seg_logits)
            batch_metrics = metrics(pred_probs, seg_labels)
            for key, value in batch_metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = 0.0
                val_metrics[key] += value.item()

            val_progress_bar.set_postfix({
                "Val Loss": val_loss / (batch_idx + 1),
                "Seg Loss": seg_loss.item(),
                "Domain Loss": domain_loss.item(),
            })

    val_loss /= len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    return val_loss, val_metrics