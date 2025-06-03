import os

import segmentation_models_pytorch as smp

from transformers import (
    MaskFormerConfig,
    DetrConfig,
    BeitConfig,
    MaskFormerForInstanceSegmentation,
    DetrForSegmentation,
    BeitForSemanticSegmentation,
)

from treemort.modeling.network.unet import UNet
from treemort.modeling.network.sa_unet import SelfAttentionUNet
from treemort.modeling.network.sa_unet_multiscale import MultiScaleAttentionUNet
from treemort.modeling.network.dinov2 import Dinov2ForSemanticSegmentation
from treemort.modeling.network.flair_unet import CombinedModel, PretrainedUNetModel
from treemort.modeling.network.flair_unet_small import FlairUNetSmall
from treemort.modeling.network.custom_models import (
    CustomMaskFormer,
    CustomDetr,
    CustomBeit,
)
from treemort.modeling.network.hcfnet.HCFnet import HCFnet
from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def configure_model(model_name, input_channels, output_channels, backbone, crop_size, cache_dir, id2label):
    model_choices = {
        "baseline": lambda: configure_baseline(input_channels, output_channels),
        "unet": lambda: configure_unet(input_channels, output_channels, backbone),
        "unetplusplus": lambda: configure_unetplusplus(input_channels, output_channels, backbone),
        "fpn": lambda: configure_fpn(input_channels, output_channels, backbone),
        "pspnet": lambda: configure_pspnet(input_channels, output_channels, backbone),
        "sa_unet": lambda: configure_sa_unet(input_channels, output_channels),
        "sa_unet_multiscale": lambda: configure_sa_unet_multiscale(input_channels, output_channels),
        "deeplabv3": lambda: configure_deeplabv3(input_channels, output_channels, backbone),
        "deeplabv3plus": lambda: configure_deeplabv3plus(input_channels, output_channels, backbone),
        "dinov2": lambda: configure_dinov2(backbone, id2label),
        "maskformer": lambda: configure_maskformer(backbone, cache_dir, id2label),
        "detr": lambda: configure_detr(backbone, cache_dir, id2label),
        "beit": lambda: configure_beit(backbone, cache_dir, id2label),
        "flair_unet": lambda: configure_flair_unet(input_channels, output_channels, crop_size),
        "flair_unet_small": lambda: configure_flair_unet_small(input_channels, output_channels, crop_size),
        "hcfnet": lambda: configure_hcfnet(input_channels, output_channels),
    }

    assert model_name in model_choices, f"[ERROR] Invalid model: {model_name}."

    model = model_choices[model_name]()
    logger.info(f"{model_name} model configured.")
    return model


def configure_baseline(input_channels, output_channels):
    model = UNet(
        in_channels=input_channels,
        n_classes=output_channels,
        padding=True,
    )
    return model


def configure_unet(input_channels, output_channels, backbone):
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=input_channels,
        classes=output_channels,
        activation=None,
    )
    return model


def configure_unetplusplus(input_channels, output_channels, backbone):
    model = smp.UnetPlusPlus(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=input_channels,
        classes=output_channels,
        activation=None,
    )
    return model


def configure_fpn(input_channels, output_channels, backbone):
    model = smp.FPN(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=input_channels,
        classes=output_channels,
        activation=None,
    )
    return model


def configure_sa_unet(input_channels, output_channels):
    model = SelfAttentionUNet(
        in_channels=input_channels,
        n_classes=output_channels,
        depth=4,
        wf=6,
        batch_norm=True,
    )
    return model


def configure_sa_unet_multiscale(input_channels, output_channels):
    model = MultiScaleAttentionUNet(
        in_channels=input_channels,
        n_classes=output_channels,
        depth=4,
        wf=6,
        batch_norm=True,
    )
    return model


def configure_deeplabv3(input_channels, output_channels, backbone):
    model = smp.DeepLabV3(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=input_channels,
        classes=output_channels,
        activation=None,
    )
    return model


def configure_deeplabv3plus(input_channels, output_channels, backbone):
    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=input_channels,
        classes=output_channels,
        activation=None,
    )
    return model


def configure_pspnet(input_channels, output_channels, backbone):
    model = smp.PSPNet(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=input_channels,
        classes=output_channels,
        activation=None,
    )
    return model


def configure_dinov2(backbone, id2label):
    model = Dinov2ForSemanticSegmentation.from_pretrained(
        backbone,
        id2label=id2label,
        num_labels=len(id2label),
    )
    return model


def configure_maskformer(backbone, cache_dir, id2label):
    # cache_dir = (
    #     conf.cache_dir
    #     if hasattr(conf, 'cache_dir') and conf.cache_dir is not None
    #     else os.environ.get("TRANSFORMERS_CACHE")
    # )

    config = MaskFormerConfig.from_pretrained(
        backbone,
        num_labels=len(id2label),
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    model = CustomMaskFormer(config)
    pretrained_model = MaskFormerForInstanceSegmentation.from_pretrained(
        backbone, cache_dir=cache_dir, local_files_only=True
    )
    model.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
    return model


def configure_detr(backbone, cache_dir, id2label):
    # cache_dir = (
    #     conf.cache_dir
    #     if hasattr(conf, 'cache_dir') and conf.cache_dir is not None
    #     else os.environ.get("TRANSFORMERS_CACHE")
    # )

    config = DetrConfig.from_pretrained(
        backbone,
        num_labels=len(id2label),
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    model = CustomDetr(config)

    pretrained_model = DetrForSegmentation.from_pretrained(
        backbone,
        num_labels=len(id2label),
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
        local_files_only=True,
    )

    state_dict = pretrained_model.detr.state_dict()
    del state_dict["class_labels_classifier.weight"]
    del state_dict["class_labels_classifier.bias"]

    model.detr.load_state_dict(state_dict, strict=False)
    return model


def configure_beit(backbone, cache_dir, id2label):
    # # Use conf.cache_dir if defined; otherwise, fall back to TRANSFORMERS_CACHE
    # cache_dir = (
    #     conf.cache_dir
    #     if hasattr(conf, 'cache_dir') and conf.cache_dir is not None
    #     else os.environ.get("TRANSFORMERS_CACHE")
    # )

    config = BeitConfig.from_pretrained(
        backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True, cache_dir=cache_dir
    )
    model = CustomBeit(config)
    pretrained_model = BeitForSemanticSegmentation.from_pretrained(
        backbone, cache_dir=cache_dir, local_files_only=True
    )
    model.beit.load_state_dict(pretrained_model.beit.state_dict(), strict=False)
    return model


def configure_flair_unet(input_channels, output_channels, crop_size):
    repo_id = "IGNF/FLAIR-INC_rgbi_15cl_resnet34-unet"
    filename = "FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth"

    pretrained_model = PretrainedUNetModel(
        repo_id=repo_id,
        filename=filename,
        architecture="unet",
        encoder="resnet34",
        n_channels=input_channels,
        n_classes=15,
        use_metadata=False,
    ).get_model()

    model = CombinedModel(
        pretrained_model=pretrained_model,
        n_classes=output_channels,
        output_size=crop_size,
    )
    return model


def configure_flair_unet_small(input_channels, output_channels, crop_size):
    model = FlairUNetSmall(
        in_channels=input_channels,
        out_channels=output_channels,
        base_ch=32,
        crop_size=crop_size
    )
    return model


def configure_hcfnet(input_channels, output_channels):
    model = HCFnet(input_channels, output_channels)
    return model
