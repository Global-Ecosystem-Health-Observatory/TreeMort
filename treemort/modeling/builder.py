import torch

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.checkpoints import get_checkpoint

logger = get_logger(__name__)


def resume_or_load(conf, id2label, n_batches, device):
    logger.info("Building student and teacher models...")

    student_model, teacher_model, optimizer, criterion, metrics = build_model(conf, id2label, device)

    callbacks = build_callbacks(n_batches, conf.output_dir, optimizer)

    if conf.teacher_model_name:
        load_teacher_weights(teacher_model, conf, device)

    if conf.resume:
        load_checkpoint(student_model, conf, device)
    else:
        logger.info("Training student model from scratch.")

    return student_model, teacher_model, optimizer, criterion, metrics, callbacks


def load_teacher_weights(teacher_model, conf, device):
    checkpoint_path = get_checkpoint(conf.output_dir, model_type="teacher", teacher_model_name=conf.teacher_model_name)

    if checkpoint_path:
        teacher_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info(f"Loaded teacher model weights from {checkpoint_path}.")
    else:
        raise FileNotFoundError("Teacher model checkpoint not found.")


def load_checkpoint(model, conf, device):
    checkpoint_path = get_checkpoint(conf.output_dir, model_weights=conf.model_weights, model_type="student")

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info(f"Loaded student model weights from {checkpoint_path}.")
    else:
        logger.info("No student model checkpoint found. Training from scratch.")


def build_model(conf, id2label, device):
    student_model = configure_model(conf, id2label)
    student_model.to(device)
    logger.info("Student model successfully moved to device.")

    teacher_model = configure_model(conf, id2label)
    teacher_model.to(device)
    logger.info("Teacher model successfully moved to device.")

    optimizer = configure_optimizer(student_model, conf.learning_rate)
    criterion, metrics = configure_loss_and_metrics(conf)

    return student_model, teacher_model, optimizer, criterion, metrics