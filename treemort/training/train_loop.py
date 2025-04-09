import torch
from tqdm import tqdm

import torch.nn.functional as F

from treemort.training.output_processing import process_model_output


def train_one_epoch(
    loss_fn,
    update_fn,
    student_model,
    teacher_model,
    optimizer,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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
    train_metrics = {}

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)

    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        loss, student_logits, teacher_logits = loss_fn(
            student_model,
            teacher_model,
            images,
            labels,
            conf,
            criterion,
            kd_criterion,
            class_weights=class_weights,
            **kwargs,
        )
        loss.backward()
        optimizer.step()

        if update_fn is not None:
            update_fn(student_model, teacher_model, **kwargs)

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


def loss_fn_basic(
    student_model,
    teacher_model,
    images,
    labels,
    conf,
    criterion,
    kd_criterion,
    class_weights,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    student_logits = process_model_output(student_model, images, conf)
    with torch.no_grad():
        teacher_logits = process_model_output(teacher_model, images, conf)

    loss_standard = criterion(student_logits, labels, class_weights=class_weights)

    loss_distillation = kd_criterion(
        torch.sigmoid(student_logits / temperature),
        torch.sigmoid(teacher_logits / temperature),
    )

    loss = alpha * loss_distillation + (1 - alpha) * loss_standard

    return loss, student_logits, teacher_logits


def loss_fn_self(
    student_model,
    ema_model,
    images,
    labels,
    conf,
    criterion,
    kd_criterion,
    class_weights,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    student_logits = process_model_output(student_model, images, conf)
    with torch.no_grad():
        teacher_logits = process_model_output(ema_model, images, conf)

    loss_standard = criterion(student_logits, labels, class_weights=class_weights)

    loss_distillation = kd_criterion(
        torch.sigmoid(student_logits / temperature),
        torch.sigmoid(teacher_logits / temperature),
    )

    loss = alpha * loss_distillation + (1 - alpha) * loss_standard
    
    return loss, student_logits, teacher_logits


def loss_fn_feature(
    student_model,
    teacher_model,
    images,
    labels,
    conf,
    criterion,
    kd_criterion,
    class_weights,
    alpha=0.5,
    temperature=2.0,
    lambda_feature=1.0,
    **kwargs,
):
    student_output, student_features = student_model(images)
    with torch.no_grad():
        teacher_output, teacher_features = teacher_model(images)

    loss_standard = criterion(student_output, labels, class_weights=class_weights)

    student_probs = torch.sigmoid(student_output / temperature)
    teacher_probs = torch.sigmoid(teacher_output / temperature)

    loss_distillation = kd_criterion(student_probs.log(), teacher_probs)

    loss_feature = sum(
        F.mse_loss(s_feat, t_feat)
        for s_feat, t_feat in zip(student_features, teacher_features)
    ) / len(student_features)
    
    loss = loss_standard + alpha * loss_distillation + lambda_feature * loss_feature

    return loss, student_output, teacher_output


def loss_fn_ensemble(
    student_model,
    teacher_models,
    images,
    labels,
    conf,
    criterion,
    kd_criterion,
    class_weights,
    alpha=0.5,
    temperature=2.0,
    **kwargs,
):
    student_logits = process_model_output(student_model, images, conf)

    teacher_predictions = []
    for teacher in teacher_models:
        with torch.no_grad():
            teacher_logits = process_model_output(teacher, images, conf)
            teacher_predictions.append(teacher_logits)

    ensemble_logits = torch.mean(torch.stack(teacher_predictions), dim=0)
    
    loss_standard = criterion(student_logits, labels, class_weights=class_weights)
    
    student_probs = torch.sigmoid(student_logits / temperature)
    ensemble_probs = torch.sigmoid(ensemble_logits / temperature)
    loss_distillation = kd_criterion(student_probs.log(), ensemble_probs)
    
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
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        conf,
        device,
        alpha=alpha,
        temperature=temperature,
        **kwargs,
    )


def train_one_epoch_self_distillation(
    student_model,
    ema_model,
    optimizer,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        conf,
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
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        conf,
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
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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
        criterion,
        kd_criterion,
        metrics,
        train_loader,
        conf,
        device,
        alpha=alpha,
        temperature=temperature,
        **kwargs,
    )


def train_one_epoch_self_distillation(
    student_model,
    ema_model,
    optimizer,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)

    train_progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        student_logits = process_model_output(student_model, images, conf)

        with torch.no_grad():
            teacher_logits = process_model_output(ema_model, images, conf)

        loss_standard = criterion(student_logits, labels, class_weights=class_weights)

        loss_distillation = kd_criterion(
            torch.sigmoid(student_logits / temperature),
            torch.sigmoid(teacher_logits / temperature),
        )

        loss = alpha * loss_distillation + (1 - alpha) * loss_standard

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for ema_param, student_param in zip(
                ema_model.parameters(), student_model.parameters()
            ):
                ema_param.data = beta * ema_param.data + (1 - beta) * student_param.data

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


def train_one_epoch_feature_level_distillation(
    student_model,
    teacher_model,
    optimizer,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)

    train_progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        student_output, student_features = student_model(images)

        with torch.no_grad():
            teacher_output, teacher_features = teacher_model(images)

        loss_standard = criterion(student_output, labels, class_weights=class_weights)

        student_probs = torch.sigmoid(student_output / temperature)
        teacher_probs = torch.sigmoid(teacher_output / temperature)
        loss_distillation = kd_criterion(student_probs.log(), teacher_probs)

        loss_feature = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            loss_feature += F.mse_loss(s_feat, t_feat)
        loss_feature /= len(student_features)

        loss = loss_standard + alpha * loss_distillation + lambda_feature * loss_feature

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred_probs = torch.sigmoid(student_output)

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


def train_one_epoch_ensemble_distillation(
    student_model,
    teacher_models,
    optimizer,
    criterion,
    kd_criterion,
    metrics,
    train_loader,
    conf,
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

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)

    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        student_logits = process_model_output(student_model, images, conf)

        teacher_predictions = []
        for teacher in teacher_models:
            with torch.no_grad():  # No gradient computation for teachers
                teacher_logits = process_model_output(teacher, images, conf)
                teacher_predictions.append(teacher_logits)

        ensemble_logits = torch.mean(torch.stack(teacher_predictions), dim=0)

        loss_standard = criterion(student_logits, labels, class_weights=class_weights)

        student_probs = torch.sigmoid(student_logits / temperature)
        ensemble_probs = torch.sigmoid(ensemble_logits / temperature)
        loss_distillation = kd_criterion(student_probs.log(), ensemble_probs)

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
