import torch
from tqdm import tqdm

from treemort.training.output_processing import process_model_output


def train_one_epoch_kd(student_model, teacher_model, optimizer, criterion, kd_criterion, metrics, train_loader, conf, device, alpha=0.5, temperature=2.0):
    student_model.train()
    teacher_model.eval()
    
    train_loss = 0.0
    train_metrics = {}

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)
    
    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        student_logits = process_model_output(student_model, images, conf)

        with torch.no_grad():
            teacher_logits = process_model_output(teacher_model, images, conf)

        loss_standard = criterion(student_logits, labels, class_weights=class_weights)

        loss_distillation = kd_criterion(
            torch.sigmoid(student_logits / temperature), 
            torch.sigmoid(teacher_logits / temperature)
        )

        loss = alpha * loss_distillation + (1 - alpha) * loss_standard

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred_probs = torch.sigmoid(student_logits)

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