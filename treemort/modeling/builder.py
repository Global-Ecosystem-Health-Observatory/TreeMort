import os
import torch

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.checkpoints import get_checkpoint

logger = get_logger(__name__)


def resume_or_load(conf, id2label, n_batches, device):
    logger.info("Building student and teacher models...")

    student_model, teacher_model, optimizer, schedular, criterion, metrics = build_model(conf, id2label, device, total_steps=conf.epochs * n_batches)

    callbacks = build_callbacks(n_batches, os.path.join(conf.output_dir, conf.model), optimizer, "best.weights."+ conf.country +"."+ conf.distillation_method + "_nd.pth")

    if conf.teacher_model_file_names:
        load_teacher_weights(teacher_model, conf, device)

    if conf.resume:
        load_checkpoint_if_available(student_model, conf, device)
    else:
        logger.info("Training student model from scratch.")

    return student_model, teacher_model, optimizer, schedular, criterion, metrics, callbacks


def load_teacher_weights(teacher_model, conf, device):
    # If teacher_model is a list, load weights for each teacher
    if isinstance(teacher_model, list):
        # Ensure teacher_model_names and teacher_model_file_name are lists
        if not isinstance(conf.teacher_model_names, list):
            conf.teacher_model_names = [conf.teacher_model_names]
        if not isinstance(conf.teacher_model_file_names, list):
            conf.teacher_model_file_names = [conf.teacher_model_file_names]
        for t_model, t_name, t_file in zip(teacher_model, conf.teacher_model_names, conf.teacher_model_file_names):
            checkpoint_path = get_checkpoint(conf.output_dir, model_name=t_name, model_file_name=t_file)
            if checkpoint_path:
                t_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
                logger.info(f"Loaded teacher model weights for {t_name} from {checkpoint_path}.")
            else:
                raise FileNotFoundError(f"Teacher model checkpoint not found for {t_name}.")
    else:
        # Single teacher model case
        t_name = conf.teacher_model_names[0] if isinstance(conf.teacher_model_names, list) else conf.teacher_model_names
        t_file = conf.teacher_model_file_names[0] if isinstance(conf.teacher_model_file_names, list) else conf.teacher_model_file_names
        checkpoint_path = get_checkpoint(conf.output_dir, model_name=t_name, model_file_name=t_file)
        if checkpoint_path:
            teacher_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            logger.info(f"Loaded teacher model weights for {t_name} from {checkpoint_path}.")
        else:
            raise FileNotFoundError("Teacher model checkpoint not found.")


def load_checkpoint_if_available(model, conf, device):
    checkpoint_path = get_checkpoint(conf.output_dir, model_name=conf.model, model_file_name=conf.best_model)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        logger.info(f"Loaded student model weights from {checkpoint_path}.")
    else:
        logger.info("No student model checkpoint found. Training from scratch.")


def build_model(conf, id2label, device, total_steps=1):
    student_model = configure_model(conf.model, conf.input_channels, conf.output_channels, conf.backbone, conf.test_crop_size, conf.cache_dir, id2label)
    student_model.to(device)
    logger.info("Student model successfully moved to device.")

    if hasattr(conf, 'teacher_model_names') and isinstance(conf.teacher_model_names, list):
        teacher_model = []
        for model_name, backbone in zip(conf.teacher_model_names, conf.teacher_backbones):
            t_model = configure_model(model_name, conf.input_channels, conf.output_channels, backbone, conf.test_crop_size, conf.cache_dir, id2label)
            t_model.to(device)
            teacher_model.append(t_model)
        logger.info(f"{len(teacher_model)} teacher models successfully moved to device.")
    else:
        teacher_model = configure_model(conf.teacher_model_names, conf.input_channels, conf.output_channels, conf.teacher_backbones, conf.test_crop_size, conf.cache_dir, id2label)
        teacher_model.to(device)
        logger.info("Teacher model successfully moved to device.")

    optimizer, schedular = configure_optimizer(student_model, conf.learning_rate, total_steps)
    criterion, metrics = configure_loss_and_metrics(conf)

    return student_model, teacher_model, optimizer, schedular, criterion, metrics
