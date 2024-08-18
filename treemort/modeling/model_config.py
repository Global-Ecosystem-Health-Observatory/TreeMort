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
from treemort.modeling.network.flair_unet import CombinedModel, PretrainedUNetModel
from treemort.modeling.network.custom_models import (
    CustomMaskFormer,
    CustomDetr,
    CustomBeit,
)


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
    }

    assert conf.model in model_choices, f"[ERROR] Invalid model: {conf.model}."

    model = model_choices[conf.model]()
    print(f"[INFO] {conf.model} model configured.")
    return model


def configure_unet(conf):
    model = smp.Unet(encoder_name="resnet34", in_channels=conf.input_channels, classes=conf.output_channels, activation=None,)
    print("[INFO] Unet model configured with pre-trained weights.")
    return model


def configure_sa_unet(conf):
    model = SelfAttentionUNet(in_channels=conf.input_channels, n_classes=conf.output_channels, depth=4, wf=6, batch_norm=True,)
    print("[INFO] SA-Unet model configured with pre-trained weights.")
    return model


def configure_sa_unet_multiscale(conf):
    model = MultiScaleAttentionUNet(in_channels=conf.input_channels, n_classes=conf.output_channels, depth=4, wf=6, batch_norm=True,)
    print("[INFO] Multiscale SA-Unet multi model configured with pre-trained weights.")
    return model


def configure_deeplabv3_plus(conf):
    model = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=conf.input_channels, encoder_weights="imagenet",)
    print("[INFO] Deeplabv3+ model configured with pre-trained weights.")
    return model


def configure_dinov2(conf, id2label):
    model = Dinov2ForSemanticSegmentation.from_pretrained(conf.backbone, id2label=id2label, num_labels=len(id2label),)
    print("[INFO] DINOv2 model configured with pre-trained weights.")
    return model


def configure_maskformer(conf, id2label):
    config = MaskFormerConfig.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)
    model = CustomMaskFormer(config)
    pretrained_model = MaskFormerForInstanceSegmentation.from_pretrained(conf.backbone)
    model.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
    print("[INFO] MaskFormer model configured with pre-trained weights.")
    return model


def configure_detr(conf, id2label):
    config = DetrConfig.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)
    model = CustomDetr(config)
    
    pretrained_model = DetrForSegmentation.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)

    state_dict = pretrained_model.detr.state_dict()
    del state_dict["class_labels_classifier.weight"]
    del state_dict["class_labels_classifier.bias"]

    model.detr.load_state_dict(state_dict, strict=False)
    print("[INFO] DETR model configured with pre-trained weights.")
    return model


def configure_beit(conf, id2label):
    config = BeitConfig.from_pretrained(conf.backbone, num_labels=len(id2label), id2label=id2label, ignore_mismatched_sizes=True,)
    model = CustomBeit(config)
    pretrained_model = BeitForSemanticSegmentation.from_pretrained(conf.backbone)
    model.beit.load_state_dict(pretrained_model.beit.state_dict(), strict=False)
    print("[INFO] BEiT model configured with pre-trained weights.")
    return model


def configure_flair_unet(conf):
    repo_id = "IGNF/FLAIR-INC_rgbi_15cl_resnet34-unet"
    filename = "FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth"

    pretrained_model = PretrainedUNetModel(
        repo_id=repo_id,
        filename=filename,
        architecture="unet",
        encoder="resnet34",
        n_channels=4,
        n_classes=15,
        use_metadata=False,
    ).get_model()

    model = CombinedModel(
        pretrained_model=pretrained_model,
        n_classes=1,
        output_size=conf.test_crop_size,
    )
    print("[INFO] FLAIR-UNet model configured with pre-trained weights.")
    return model
