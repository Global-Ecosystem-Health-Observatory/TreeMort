import torch
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict

from treemort.training.output_processing import process_model_output


def train_one_epoch(
    loss_fn,
    update_fn,
    student_model,
    teacher_model,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_names,
    device,
    **kwargs,
):
    student_model.train()
    if isinstance(teacher_model, list):
        for t in teacher_model:
            t.eval()
    else:
        teacher_model.eval()

    train_loss = 0.0
    train_metrics = defaultdict(float)

    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        loss, student_logits, teacher_logits = loss_fn(
            student_model,
            teacher_model,
            images,
            labels,
            model_name,
            teacher_model_names,
            criterion,
            kd_criterion,
            **kwargs,
        )

        loss.backward()
        optimizer.step()
        
        scheduler.step()

        with torch.no_grad():
            batch_metrics = metrics(student_logits, labels)

        if update_fn is not None:
            update_fn(student_model, teacher_model, **kwargs)

        train_loss += loss.item()

        batch_metrics = metrics(student_logits, labels)

        for key, value in batch_metrics.items():
            if key not in train_metrics:
                train_metrics[key] = 0.0
            train_metrics[key] += value.item()

        train_progress_bar.set_postfix({"Train Loss": train_loss / (batch_idx + 1)})

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, train_metrics


def loss_fn_basic(
    student_model,
    teacher_model,
    images,
    labels,
    model_name,
    teacher_model_name,
    criterion,
    kd_criterion,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    student_logits, _ = process_model_output(student_model, images, model_name)
    with torch.no_grad():
        teacher_logits_list = []
        for aug_fn, rev_fn in [
            (lambda x: x, lambda x: x),
            (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
            (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2]))
        ]:
            images_aug = aug_fn(images)
            with torch.no_grad():
                logits_aug, _ = process_model_output(teacher_model, images_aug, teacher_model_name)
            teacher_logits_list.append(rev_fn(logits_aug))

        teacher_logits = torch.mean(torch.stack(teacher_logits_list), dim=0)

    loss_standard = criterion(student_logits, labels)

    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)

    # Optional sharpening before clamping
    sharpen_temperature = kwargs.get("sharpen_temperature", temperature)
    teacher_probs = torch.pow(teacher_probs, 1.0 / sharpen_temperature)
    teacher_probs = torch.clamp(teacher_probs, min=0.05, max=0.95)

    foreground_mask = (labels[:, 0:1, :, :] > 0).float()
    background_mask = 1.0 - foreground_mask

    w_fg = kwargs.get("foreground_weight", 5.0)
    w_bg = kwargs.get("background_weight", 1.0)

    weight_map = foreground_mask * w_fg + background_mask * w_bg

    confidence_mask = (teacher_probs > 0.3).float()
    teacher_probs = teacher_probs * confidence_mask
    student_probs = student_probs * confidence_mask
    weight_map = weight_map * confidence_mask

    confidence_weights = torch.clamp((teacher_probs - 0.3) / 0.7, 0, 1)  # Range 0 to 1
    loss_distillation = F.binary_cross_entropy(
        student_probs,
        teacher_probs,
        weight=weight_map * confidence_weights
    ) * (temperature ** 2)

    loss = alpha * loss_distillation + (1 - alpha) * loss_standard

    return loss, student_logits, teacher_logits


def loss_fn_self(
    student_model,
    ema_model,
    images,
    labels,
    model_name,
    teacher_model_name,
    criterion,
    kd_criterion,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    student_logits, _ = process_model_output(student_model, images, model_name)
    with torch.no_grad():
        teacher_logits, _ = process_model_output(ema_model, images, teacher_model_name)

    loss_standard = criterion(student_logits, labels)

    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)

    # Optional sharpening
    sharpen_temperature = kwargs.get("sharpen_temperature", temperature)
    teacher_probs = torch.pow(teacher_probs, 1.0 / sharpen_temperature)
    teacher_probs = torch.clamp(teacher_probs, min=0.05, max=0.95)

    foreground_mask = (labels[:, 0:1, :, :] > 0).float()
    background_mask = 1.0 - foreground_mask

    w_fg = kwargs.get("foreground_weight", 5.0)
    w_bg = kwargs.get("background_weight", 1.0)
    weight_map = foreground_mask * w_fg + background_mask * w_bg

    confidence_mask = (teacher_probs > 0.3).float()
    teacher_probs = teacher_probs * confidence_mask
    student_probs = student_probs * confidence_mask
    weight_map = weight_map * confidence_mask

    confidence_weights = torch.clamp((teacher_probs - 0.3) / 0.7, 0, 1)

    loss_distillation = F.binary_cross_entropy(
        student_probs,
        teacher_probs,
        weight=weight_map * confidence_weights
    ) * (temperature ** 2)

    loss = alpha * loss_distillation + (1 - alpha) * loss_standard
    return loss, student_logits, teacher_logits


def loss_fn_feature(
    student_model,
    teacher_model,
    images,
    labels,
    model_name,
    teacher_model_name,
    criterion,
    kd_criterion,
    alpha=0.5,
    temperature=2.0,
    lambda_feature=1.0,
    **kwargs,
):
    student_logits, student_features = process_model_output(student_model, images, model_name)
    with torch.no_grad():
        teacher_logits, teacher_features = process_model_output(teacher_model, images, teacher_model_name)

    loss_standard = criterion(student_logits, labels)

    loss_distillation = kd_criterion(
        F.logsigmoid(student_logits / temperature),
        torch.sigmoid(teacher_logits / temperature),
    )

    loss_feature = sum(
        F.mse_loss(s_feat, t_feat)
        for s_feat, t_feat in zip(student_features, teacher_features)
    ) / len(student_features)
    
    loss = loss_standard + alpha * loss_distillation + lambda_feature * loss_feature

    return loss, student_logits, teacher_logits


def loss_fn_ensemble(
    student_model,
    teacher_models,
    images,
    labels,
    model_name,
    teacher_model_names,
    criterion,
    kd_criterion,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    student_logits, _ = process_model_output(student_model, images, model_name)

    teacher_predictions = []
    for teacher, teacher_name in zip(teacher_models, teacher_model_names):
        with torch.no_grad():
            teacher_logits, _ = process_model_output(teacher, images, teacher_name)
            teacher_predictions.append(teacher_logits)

    ensemble_logits = torch.mean(torch.stack(teacher_predictions), dim=0)
    
    loss_standard = criterion(student_logits, labels)

    loss_distillation = kd_criterion(
        F.logsigmoid(student_logits / temperature),
        torch.sigmoid(ensemble_logits / temperature),
    )
    
    loss = alpha * loss_distillation + (1 - alpha) * loss_standard
    
    return loss, student_logits, ensemble_logits


def update_fn_self(student_model, ema_model, beta=0.999, **kwargs):
    with torch.no_grad():
        for ema_param, student_param in zip(
            ema_model.parameters(), student_model.parameters()
        ):
            ema_param.data = beta * ema_param.data + (1 - beta) * student_param.data


def train_one_epoch_distillation(
    student_model,
    teacher_model,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_name,
    device,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    return train_one_epoch(
        loss_fn_basic,
        None,
        student_model,
        teacher_model,
        optimizer,
        scheduler,
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        model_name,
        teacher_model_name,
        device,
        alpha=alpha,
        temperature=temperature,
        **kwargs,
    )


def train_one_epoch_self_distillation(
    student_model,
    ema_model,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_name,
    device,
    alpha=0.5,
    temperature=2.0,
    beta=0.999,
    **kwargs,
):
    return train_one_epoch(
        loss_fn_self,
        update_fn_self,
        student_model,
        ema_model,
        optimizer,
        scheduler,
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        model_name,
        teacher_model_name,
        device,
        alpha=alpha,
        temperature=temperature,
        beta=beta,
        **kwargs,
    )


def train_one_epoch_feature_level_distillation(
    student_model,
    teacher_model,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_name,
    device,
    alpha=0.5,
    temperature=2.0,
    lambda_feature=1.0,
    **kwargs,
):
    return train_one_epoch(
        loss_fn_feature,
        None,
        student_model,
        teacher_model,
        optimizer,
        scheduler,
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        model_name,
        teacher_model_name,
        device,
        alpha=alpha,
        temperature=temperature,
        lambda_feature=lambda_feature,
        **kwargs,
    )


def train_one_epoch_ensemble_distillation(
    student_model,
    teacher_models,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_names,
    device,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    return train_one_epoch(
        loss_fn_ensemble,
        None,
        student_model,
        teacher_models,
        optimizer,
        scheduler,
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        model_name,
        teacher_model_names,
        device,
        alpha=alpha,
        temperature=temperature,
        **kwargs,
    )


def train_one_epoch_self_distillation(
    student_model,
    ema_model,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_name,
    device,
    alpha=0.5,
    temperature=2.0,
    beta=0.999,
    **kwargs,
):
    student_model.train()
    ema_model.eval()

    train_loss = 0.0
    train_metrics = {}

    train_progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        student_logits, _ = process_model_output(student_model, images, model_name)

        with torch.no_grad():
            teacher_logits, _ = process_model_output(ema_model, images, teacher_model_name)

        loss_standard = criterion(student_logits, labels)

        loss_distillation = kd_criterion(
            F.logsigmoid(student_logits / temperature),
            torch.sigmoid(teacher_logits / temperature),
        )

        loss = alpha * loss_distillation + (1 - alpha) * loss_standard

        loss.backward()
        optimizer.step()

        scheduler.step()

        with torch.no_grad():
            for ema_param, student_param in zip(
                ema_model.parameters(), student_model.parameters()
            ):
                ema_param.data = beta * ema_param.data + (1 - beta) * student_param.data

        train_loss += loss.item()

        batch_metrics = metrics(student_logits, labels)

        for key, value in batch_metrics.items():
            if key not in train_metrics:
                train_metrics[key] = 0.0
            train_metrics[key] += value.item()

        train_progress_bar.set_postfix({"Train Loss": train_loss / (batch_idx + 1)})

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, train_metrics


def train_one_epoch_feature_level_distillation(
    student_model,
    teacher_model,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_name,
    device,
    alpha=0.5,
    temperature=2.0,
    lambda_feature=1.0,
    **kwargs,
):
    student_model.train()
    teacher_model.eval()

    train_loss = 0.0
    train_metrics = {}

    train_progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        student_logits, student_features = process_model_output(student_model, images, model_name)

        with torch.no_grad():
            teacher_logits, teacher_features = process_model_output(teacher_model, images, teacher_model_name)

        loss_standard = criterion(student_logits, labels)

        loss_distillation = kd_criterion(
            F.logsigmoid(student_logits / temperature),
            torch.sigmoid(teacher_logits / temperature),
        )

        loss_feature = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            loss_feature += F.mse_loss(s_feat, t_feat)
        loss_feature /= len(student_features)

        loss = loss_standard + alpha * loss_distillation + lambda_feature * loss_feature

        loss.backward()
        optimizer.step()

        scheduler.step()

        train_loss += loss.item()

        batch_metrics = metrics(student_logits, labels)
        
        for key, value in batch_metrics.items():
            if key not in train_metrics:
                train_metrics[key] = 0.0
            train_metrics[key] += value.item()

        train_progress_bar.set_postfix({"Train Loss": train_loss / (batch_idx + 1)})

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, train_metrics


def train_one_epoch_ensemble_distillation(
    student_model,
    teacher_models,
    optimizer,
    scheduler,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    model_name,
    teacher_model_names,
    device,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    student_model.train()
    for teacher in teacher_models:
        teacher.eval()

    train_loss = 0.0
    train_metrics = {}

    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        student_logits, _ = process_model_output(student_model, images, model_name)

        teacher_predictions = []
        for teacher, teacher_name in zip(teacher_models, teacher_model_names):
            with torch.no_grad():  # No gradient computation for teachers
                teacher_logits, _ = process_model_output(teacher, images, teacher_name)
                teacher_predictions.append(teacher_logits)

        ensemble_logits = torch.mean(torch.stack(teacher_predictions), dim=0)

        loss_standard = criterion(student_logits, labels)

        loss_distillation = kd_criterion(
            F.logsigmoid(student_logits / temperature),
            torch.sigmoid(ensemble_logits / temperature),
        )
        
        loss = alpha * loss_distillation + (1 - alpha) * loss_standard

        loss.backward()
        optimizer.step()

        scheduler.step()

        train_loss += loss.item()

        batch_metrics = metrics(student_logits, labels)

        for key, value in batch_metrics.items():
            if key not in train_metrics:
                train_metrics[key] = 0.0
            train_metrics[key] += value.item() if torch.is_tensor(value) else value

        train_progress_bar.set_postfix({
            "Loss": f"{train_loss/(batch_idx+1):.4f}",
            "IOU": f"{train_metrics.get('iou_segments',0)/(batch_idx+1):.4f}",
            "F1": f"{train_metrics.get('f_score_segments',0)/(batch_idx+1):.4f}"
        })

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, dict(train_metrics)