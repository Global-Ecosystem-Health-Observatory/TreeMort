from treemort.training.train_loop import train_one_epoch
from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks
from treemort.utils.callbacks import EarlyStopping


def trainer(
    model,
    optimizer,
    criterion,
    metrics,
    train_loader,
    val_loader,
    conf,
    callbacks,
    image_processor=None,
    class_weights=None,
):
    device = next(model.parameters()).device

    class_weights = class_weights.to(device)

    for epoch in range(conf.epochs):
        print(f"[INFO] Epoch {epoch + 1}/{conf.epochs} - Training started.")

        train_loss, train_metrics = train_one_epoch(model, optimizer, criterion, metrics, train_loader, conf, device, image_processor, class_weights)

        print(f"[INFO] Epoch {epoch + 1} - Training completed.")
        print(f"[INFO] Training Loss: {train_loss:.4f}")
        print(f"[INFO] Training Metrics: {train_metrics}")

        val_loss, val_metrics = validate_one_epoch(model, criterion, metrics, val_loader, conf, device, image_processor, class_weights)

        print(f"[INFO] Epoch {epoch + 1} - Validation completed.")
        print(f"[INFO] Validation Loss: {val_loss:.4f}")
        print(f"[INFO] Validation Metrics: {val_metrics}")

        handle_callbacks(callbacks, epoch, model, optimizer, val_loss)

        if any([isinstance(cb, EarlyStopping) and cb.stop_training for cb in callbacks]):
            print("[INFO] Early stopping triggered.")
            break

    print("[INFO] Training process completed.")
