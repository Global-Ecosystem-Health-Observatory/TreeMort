import torch

import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from treemort.modeling.network.sa_unet import SelfAttentionUNet
from treemort.modeling.network.flair_unet import CombinedModel, PretrainedUNetModel

from treemort.utils.checkpoints import get_checkpoint
from treemort.utils.loss import hybrid_loss, mse_loss, iou_score, f_score

def resume_or_load(conf, device):

    model, optimizer, criterion, metrics = build_model(conf, device=device)

    print("[INFO] Model built.")

    if conf.resume:
        checkpoint = get_checkpoint(conf.model_weights, conf.output_dir)

        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
            print(f"[INFO] Loaded model weights from {checkpoint}")

        else:
            print(
                "[INFO] No checkpoint found. Proceeding without loading model weights."
            )

    else:
        print("[INFO] Model training from scratch.")

    return model, optimizer, criterion, metrics


def build_model(conf, device):
    assert conf.model in [
        "unet",
        "sa_unet",
        "flair_unet",
        "deeplabv3+",
    ], f"Model {conf.model} unavailable."
    assert conf.activation in [
        "tanh",
        "sigmoid",
    ], f"Model activation {conf.activation} unavailable."
    assert conf.loss in ["mse", "hybrid"], f"Model loss {conf.loss} unavailable."

    if conf.model == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            in_channels=conf.input_channels,
            classes=conf.output_channels,
            activation=None,
        )

    elif conf.model == "sa_unet":

        model = SelfAttentionUNet(
            in_channels=conf.input_channels,
            n_classes=conf.output_channels,
            depth=4,
            wf=6,
            batch_norm=True,
        )

    elif conf.model == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            backbone="resnet50", in_channels=conf.input_channels, encoder_weights="imagenet"
        )

    elif conf.model == "flair_unet":

        repo_id = "IGNF/FLAIR-INC_rgbi_15cl_resnet34-unet"
        filename = "FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth"

        pretrained_model = PretrainedUNetModel(
            repo_id=repo_id,
            filename=filename,
            architecture="unet",
            encoder="resnet34",
            n_channels=4,
            n_classes=15,
            use_metadata=False
        )

        pretrained_model = pretrained_model.get_model()

        model = CombinedModel(pretrained_model=pretrained_model, n_classes=1, output_size=conf.test_crop_size)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    if conf.loss == "hybrid":
        criterion = hybrid_loss

        def metrics(pred, target):
            return {
                "iou_score": iou_score(pred, target, conf.threshold),
                "f_score": f_score(pred, target, conf.threshold),
            }

    elif conf.loss == "mse":
        criterion = mse_loss

        def metrics(pred, target):
            mse = mse_loss(pred, target)
            mae = nn.functional.l1_loss(pred, target)
            rmse = torch.sqrt(mse)
            return {"mse": mse, "mae": mae, "rmse": rmse}

    return model, optimizer, criterion, metrics
