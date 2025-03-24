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
from treemort.modeling.network.custom_models import (
    CustomMaskFormer,
    CustomDetr,
    CustomBeit,
)
from treemort.modeling.network.hcfnet.HCFnet import HCFnet
from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def configure_model(conf, id2label):
    model_choices = {
        "baseline": lambda: configure_baseline(conf),
        "unet": lambda: configure_unet(conf),
        "unetplusplus": lambda: configure_unetplusplus(conf),
        "fpn": lambda: configure_fpn(conf),
        "pspnet": lambda: configure_pspnet(conf),
        "sa_unet": lambda: configure_sa_unet(conf),
        "sa_unet_multiscale": lambda: configure_sa_unet_multiscale(conf),
        "deeplabv3": lambda: configure_deeplabv3(conf),
        "deeplabv3plus": lambda: configure_deeplabv3plus(conf),
        "dinov2": lambda: configure_dinov2(conf, id2label),
        "maskformer": lambda: configure_maskformer(conf, id2label),
        "detr": lambda: configure_detr(conf, id2label),
        "beit": lambda: configure_beit(conf, id2label),
        "flair_unet": lambda: configure_flair_unet(conf),
        "hcfnet": lambda: configure_hcfnet(conf),
    }

    assert conf.model in model_choices, f"[ERROR] Invalid model: {conf.model}."

    model = model_choices[conf.model]()
    logger.info(f"{conf.model} model configured.")
    return model


def configure_baseline(conf):
    model = UNet(
        in_channels=conf.input_channels,
        n_classes=conf.output_channels,
    )
    return model


def configure_unet(conf):
    model = smp.Unet(
        encoder_name=conf.backbone,
        encoder_weights='imagenet',
        in_channels=conf.input_channels,
        classes=conf.output_channels,
        activation=None,
    )
    return model


def configure_unetplusplus(conf):
    model = smp.UnetPlusPlus(
        encoder_name=conf.backbone,
        encoder_weights='imagenet',
        in_channels=conf.input_channels,
        classes=conf.output_channels,
        activation=None,
    )
    return model


def configure_fpn(conf):
    model = smp.FPN(
        encoder_name=conf.backbone,
        encoder_weights='imagenet',
        in_channels=conf.input_channels,
        classes=conf.output_channels,
        activation=None,
    )
    return model


def configure_sa_unet(conf):
    model = SelfAttentionUNet(
        in_channels=conf.input_channels,
        n_classes=conf.output_channels,
        depth=4,
        wf=6,
        batch_norm=True,
    )
    return model


def configure_sa_unet_multiscale(conf):
    model = MultiScaleAttentionUNet(
        in_channels=conf.input_channels,
        n_classes=conf.output_channels,
        depth=4,
        wf=6,
        batch_norm=True,
    )
    return model


def configure_deeplabv3(conf):
    model = smp.DeepLabV3(
        encoder_name=conf.backbone,
        encoder_weights='imagenet',
        in_channels=conf.input_channels,
        classes=conf.output_channels,
        activation=None,
    )
    return model


def configure_deeplabv3plus(conf):
    model = smp.DeepLabV3Plus(
        encoder_name=conf.backbone,
        encoder_weights='imagenet',
        in_channels=conf.input_channels,
        classes=conf.output_channels,
        activation=None,
    )
    return model


def configure_pspnet(conf):
    model = smp.PSPNet(
        encoder_name=conf.backbone,
        encoder_weights='imagenet',
        in_channels=conf.input_channels,
        classes=conf.output_channels,
        activation=None,
    )
    return model


def configure_dinov2(conf, id2label):
    model = Dinov2ForSemanticSegmentation.from_pretrained(
        conf.backbone,
        id2label=id2label,
        num_labels=len(id2label),
    )
    return model


def configure_maskformer(conf, id2label):
    cache_dir = (
        conf.cache_dir
        if hasattr(conf, 'cache_dir') and conf.cache_dir is not None
        else os.environ.get("TRANSFORMERS_CACHE")
    )

    config = MaskFormerConfig.from_pretrained(
        conf.backbone,
        num_labels=len(id2label),
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    model = CustomMaskFormer(config)
    pretrained_model = MaskFormerForInstanceSegmentation.from_pretrained(
        conf.backbone, cache_dir=cache_dir, local_files_only=True
    )
    model.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
    return model


def configure_detr(conf, id2label):
    cache_dir = (
        conf.cache_dir
        if hasattr(conf, 'cache_dir') and conf.cache_dir is not None
        else os.environ.get("TRANSFORMERS_CACHE")
    )

    config = DetrConfig.from_pretrained(
        conf.backbone,
        num_labels=len(id2label),
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    model = CustomDetr(config)

    pretrained_model = DetrForSegmentation.from_pretrained(
        conf.backbone,
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


def configure_beit(conf, id2label):
    # Use conf.cache_dir if defined; otherwise, fall back to TRANSFORMERS_CACHE
    cache_dir = (
        conf.cache_dir
        if hasattr(conf, 'cache_dir') and conf.cache_dir is not None
        else os.environ.get("TRANSFORMERS_CACHE")
    )

    config = BeitConfig.from_pretrained(
        conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True, cache_dir=cache_dir
    )
    model = CustomBeit(config)
    pretrained_model = BeitForSemanticSegmentation.from_pretrained(
        conf.backbone, cache_dir=cache_dir, local_files_only=True
    )
    model.beit.load_state_dict(pretrained_model.beit.state_dict(), strict=False)
    return model


def configure_flair_unet(conf):
    repo_id = "IGNF/FLAIR-INC_rgbi_15cl_resnet34-unet"
    filename = "FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth"

    pretrained_model = PretrainedUNetModel(
        repo_id=repo_id,
        filename=filename,
        architecture="unet",
        encoder="resnet34",
        n_channels=conf.input_channels,
        n_classes=15,
        use_metadata=False,
    ).get_model()

    model = CombinedModel(
        pretrained_model=pretrained_model,
        n_classes=conf.output_channels,
        output_size=conf.test_crop_size,
    )
    return model


def configure_hcfnet(conf):
    model = HCFnet(conf.input_channels, conf.output_channels)
    return model
