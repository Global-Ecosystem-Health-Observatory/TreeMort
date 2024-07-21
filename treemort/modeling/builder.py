import os
import timm
import torch

import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from huggingface_hub import hf_hub_download

from treemort.utils.checkpoints import get_checkpoint
from treemort.modeling.network.self_attention_unet import SelfAttentionUNet
from treemort.utils.loss import hybrid_loss, mse_loss, iou_score, f_score

def create_feature_extractor(model_name, model_type, model_filename):

    checkpoint_path = hf_hub_download(repo_id=model_name, filename=model_filename)

    feature_extractor = timm.create_model(model_type, pretrained=False)
    feature_extractor.load_state_dict(torch.load(checkpoint_path))

    return feature_extractor


def resume_or_load(conf, device):

    if conf.feature_extractor == "flair":
        feature_extractor = create_feature_extractor("IGNF/FLAIR-INC_rgbi_15cl_resnet34-unet", model_type='resnet34', model_filename="FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth")
    else:
        feature_extractor = None

    model, optimizer, criterion, metrics = build_model(
        model_name=conf.model,
        input_channels=conf.input_channels,
        output_channels=conf.output_channels,
        activation=conf.activation,
        loss=conf.loss,
        learning_rate=conf.learning_rate,
        threshold=conf.threshold,
        device=device
    )

    if conf.resume:
        checkpoint = get_checkpoint(conf.model_weights, conf.output_dir)

        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
            print(f"Loaded weights from {checkpoint}")

        else:
            print("No checkpoint found. Proceeding without loading weights.")

    else:
        print("Model training from scratch.")

    return model, optimizer, criterion, metrics, feature_extractor




def build_model(
    model_name,
    input_channels,
    output_channels,
    activation,
    loss,
    learning_rate,
    threshold,
    device,
    
):
    assert model_name in [
        "unet",
        "sa_unet",
        "deeplabv3+",
    ], f"Model {model_name} unavailable."
    assert activation in [
        "tanh",
        "sigmoid",
    ], f"Model activation {activation} unavailable."
    assert loss in ["mse", "hybrid"], f"Model loss {loss} unavailable."

    if model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            in_channels=input_channels,
            classes=output_channels,
            activation=None,
        )

    elif model_name == "sa_unet":
        
        model = SelfAttentionUNet(
            in_channels=input_channels,
            n_classes=output_channels,
            depth=4,
            wf=6,
            batch_norm=True,
        )

    elif model_name == "deeplabv3+":
        model = smp.DeepLabV3Plus(backbone='resnet50', in_channels=input_channels, encoder_weights='imagenet')

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if loss == "hybrid":
        criterion = hybrid_loss

        def metrics(pred, target):
            return {
                "iou_score": iou_score(pred, target, threshold),
                "f_score": f_score(pred, target, threshold),
            }

    elif loss == "mse":
        criterion = mse_loss

        def metrics(pred, target):
            mse = mse_loss(pred, target)
            mae = nn.functional.l1_loss(pred, target)
            rmse = torch.sqrt(mse)
            return {"mse": mse, "mae": mae, "rmse": rmse}

    return model, optimizer, criterion, metrics
