import torch

import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from transformers import (
    MaskFormerForInstanceSegmentation,
    DetrForSegmentation,
    BeitForSemanticSegmentation,
)

from treemort.utils.checkpoints import get_checkpoint
from treemort.utils.loss import hybrid_loss, mse_loss, iou_score, f_score

from treemort.modeling.network.dinov2 import Dinov2ForSemanticSegmentation
from treemort.modeling.network.self_attention_unet import SelfAttentionUNet
from treemort.modeling.network.custom_models import (
    CustomMaskFormer,
    CustomDetr,
    CustomBeit,
)
from treemort.modeling.config import validate_configuration, get_model_config


def resume_or_load(conf, id2label, device):
    print("[INFO] Starting model building process.")
    model, optimizer, criterion, metrics = build_model(conf, id2label, device)

    if conf.resume:
        print("[INFO] Resuming from checkpoint.")
        checkpoint = get_checkpoint(conf.model_weights, conf.output_dir)

        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
            print(f"[INFO] Loaded weights from {checkpoint}")

        else:
            print("[INFO] No checkpoint found. Proceeding without loading weights.")

    else:
        print("[INFO] Training model from scratch.")

    return model, optimizer, criterion, metrics


def build_model(conf, id2label, device):
    print("[INFO] Validating configuration.")
    validate_configuration(conf)

    print(f"[INFO] Loading configuration for model {conf.model}.")
    config = get_model_config(conf.model, conf.backbone, len(id2label), id2label)

    print(f"[INFO] Creating model {conf.model}.")
    model = {
            "unet": create_unet_model,
            "sa_unet": create_sa_unet_model,
            "deeplabv3+": create_deeplabv3plus_model,
            "dinov2": create_dinov2_model,
            "maskformer": create_maskformer_model,
            "detr": create_detr_model,
            "beit": create_beit_model,
        }[conf.model](config=config, conf=conf, id2label=id2label, backbone=conf.backbone)

    print("[INFO] Moving model to device.")
    model.to(device)

    print("[INFO] Initializing optimizer.")
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    if conf.loss == "hybrid":
        print("[INFO] Using hybrid loss function.")
        criterion = hybrid_loss

        def metrics(pred, target):
            return {
                "iou_score": iou_score(pred, target, conf.threshold),
                "f_score": f_score(pred, target, conf.threshold),
            }

    elif conf.loss == "mse":
        print("[INFO] Using MSE loss function.")
        criterion = mse_loss

        def metrics(pred, target):
            mse = mse_loss(pred, target)
            mae = nn.functional.l1_loss(pred, target)
            rmse = torch.sqrt(mse)
            return {"mse": mse, "mae": mae, "rmse": rmse}

    print("[INFO] Model building completed.")
    return model, optimizer, criterion, metrics


def create_unet_model(conf, **kwargs):
    print("[INFO] Creating U-Net model.")
    return smp.Unet(
        encoder_name="resnet34", in_channels=conf.input_channels, classes=conf.output_channels, activation=None
    )

def create_sa_unet_model(conf, **kwargs):
    print("[INFO] Creating Self-Attention U-Net model.")
    return SelfAttentionUNet(in_channels=conf.input_channels, n_classes=conf.output_channels, depth=4, wf=6, batch_norm=True)

def create_deeplabv3plus_model(conf, **kwargs):
    print("[INFO] Creating DeepLabV3+ model.")
    return smp.DeepLabV3Plus(conf.backbone, in_channels=conf.input_channels, encoder_weights="imagenet")

def create_dinov2_model(conf, id2label, **kwargs):
    print("[INFO] Creating DINOv2 model.")
    return Dinov2ForSemanticSegmentation.from_pretrained(conf.backbone, id2label=id2label, num_labels=len(id2label))

def create_maskformer_model(config, backbone, **kwargs):
    print("[INFO] Creating MaskFormer model.")
    model = CustomMaskFormer(config)
    pretrained_model = MaskFormerForInstanceSegmentation.from_pretrained(backbone)
    model.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
    print("[INFO] MaskFormer model created and weights loaded.")
    return model

def create_detr_model(config, backbone, id2label, **kwargs):
    print("[INFO] Creating DETR model.")
    model = CustomDetr(config)
    pretrained_model = DetrForSegmentation.from_pretrained(
        backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True
    )
    state_dict = pretrained_model.detr.state_dict()
    del state_dict["class_labels_classifier.weight"]
    del state_dict["class_labels_classifier.bias"]
    model.detr.load_state_dict(state_dict, strict=False)
    print("[INFO] DETR model created and weights loaded.")
    return model

def create_beit_model(config, backbone, **kwargs):
    print("[INFO] Creating BEiT model.")
    model = CustomBeit(config)
    pretrained_model = BeitForSemanticSegmentation.from_pretrained(backbone)
    model.beit.load_state_dict(pretrained_model.beit.state_dict(), strict=False)
    print("[INFO] BEiT model created and weights loaded.")
    return model