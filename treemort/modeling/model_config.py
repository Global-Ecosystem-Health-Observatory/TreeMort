import segmentation_models_pytorch as smp

from transformers import (
    MaskFormerConfig, 
    DetrConfig, 
    BeitConfig,
    MaskFormerForInstanceSegmentation,
    DetrForSegmentation,
    BeitForSemanticSegmentation,
)

from treemort.modeling.network.sa_unet import SelfAttentionUNet
from treemort.modeling.network.sa_unet_multiscale import MultiScaleAttentionUNet
from treemort.modeling.network.dinov2 import Dinov2ForSemanticSegmentation
from treemort.modeling.network.flair_unet import CombinedModel, PretrainedUNetModel, StandardCombinedModel
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
        "unet": lambda: configure_unet(conf),
        "sa_unet": lambda: configure_sa_unet(conf),
        "sa_unet_multiscale": lambda: configure_sa_unet_multiscale(conf),
        "deeplabv3+": lambda: configure_deeplabv3_plus(conf),
        "dinov2": lambda: configure_dinov2(conf, id2label),
        "maskformer": lambda: configure_maskformer(conf, id2label),
        "detr": lambda: configure_detr(conf, id2label),
        "beit": lambda: configure_beit(conf, id2label),
        "flair_unet": lambda: configure_flair_unet(conf),
        "flair_unet_baseline": lambda: configure_flair_unet(conf, use_pretrain=False, use_attention=False, use_multi_task=False), # Baseline (Standard U-Net)
        "flair_unet_pretrained": lambda: configure_flair_unet(conf, use_pretrain=True, use_attention=False, use_multi_task=False), # Pretrained Encoder Only
        "flair_unet_attention": lambda: configure_flair_unet(conf, use_pretrain=True, use_attention=True, use_multi_task=False), # Self-Attention Modules
        "hcfnet": lambda: configure_hcfnet(conf),
    }

    assert conf.model in model_choices, f"[ERROR] Invalid model: {conf.model}."

    model = model_choices[conf.model]()
    logger.info(f"{conf.model} model configured.")
    return model


def configure_unet(conf):
    model = smp.Unet(encoder_name="resnet34", in_channels=conf.input_channels, classes=conf.output_channels, activation=None,)
    return model


def configure_sa_unet(conf):
    model = SelfAttentionUNet(in_channels=conf.input_channels, n_classes=conf.output_channels, depth=4, wf=6, batch_norm=True,)
    return model


def configure_sa_unet_multiscale(conf):
    model = MultiScaleAttentionUNet(in_channels=conf.input_channels, n_classes=conf.output_channels, depth=4, wf=6, batch_norm=True,)
    return model


def configure_deeplabv3_plus(conf):
    model = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=conf.input_channels, encoder_weights="imagenet",)
    return model


def configure_dinov2(conf, id2label):
    model = Dinov2ForSemanticSegmentation.from_pretrained(conf.backbone, id2label=id2label, num_labels=len(id2label),)
    return model


def configure_maskformer(conf, id2label):
    config = MaskFormerConfig.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)
    model = CustomMaskFormer(config)
    pretrained_model = MaskFormerForInstanceSegmentation.from_pretrained(conf.backbone)
    model.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
    return model


def configure_detr(conf, id2label):
    config = DetrConfig.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)
    model = CustomDetr(config)
    
    pretrained_model = DetrForSegmentation.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)

    state_dict = pretrained_model.detr.state_dict()
    del state_dict["class_labels_classifier.weight"]
    del state_dict["class_labels_classifier.bias"]

    model.detr.load_state_dict(state_dict, strict=False)
    return model


def configure_beit(conf, id2label):
    config = BeitConfig.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)
    model = CustomBeit(config)
    pretrained_model = BeitForSemanticSegmentation.from_pretrained(conf.backbone)
    model.beit.load_state_dict(pretrained_model.beit.state_dict(), strict=False)
    return model


# Updated configure function with toggles for variants
def configure_flair_unet(
    conf,
    use_pretrain=True,  # Toggle for pretrained FLAIR-INC
    use_attention=True,  # Toggle for self-attention in decoder
    use_multi_task=True,  # Toggle for multi-task (n_classes=3) vs single-task (n_classes=1)
):
    if not use_pretrain:
        # Use standard SMP U-Net with no pretrain (vanilla baseline)
        model = smp.Unet(
            encoder_name="resnet34",  # Match manuscript
            encoder_weights=None,  # No pretrain
            in_channels=conf.input_channels,
            classes=1 if not use_multi_task else 3,  # Single or multi-task
            activation=None,
        )
        return model

    # Pretrained case: Load FLAIR-INC as before
    repo_id = "IGNF/FLAIR-INC_rgbi_15cl_resnet34-unet"
    filename = "FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth"

    pretrained_model = PretrainedUNetModel(
        repo_id=repo_id,
        filename=filename,
        architecture="unet",
        encoder="resnet34",
        n_channels=conf.input_channels,
        n_classes=15,  # Original FLAIR classes; adapted later
        use_metadata=False,
    ).get_model()

    # Choose CombinedModel based on use_attention
    n_classes = 3 if use_multi_task else 1  # Multi-task: 3 channels; single-task: 1
    if use_attention:
        model = CombinedModel(
            pretrained_model=pretrained_model,
            n_classes=n_classes,
            output_size=conf.test_crop_size,
        )
    else:
        model = StandardCombinedModel(
            pretrained_model=pretrained_model,
            n_classes=n_classes,
            output_size=conf.test_crop_size,
        )

    return model


def configure_hcfnet(conf):
    model = HCFnet(conf.input_channels, conf.output_channels)
    return model