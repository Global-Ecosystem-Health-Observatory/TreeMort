from transformers import MaskFormerConfig, DetrConfig, BeitConfig


def validate_configuration(conf):
    assert conf.model in [
        "unet",
        "sa_unet",
        "deeplabv3+",
        "maskformer",
        "detr",
        "beit",
        "dinov2",
    ], f"Model {conf.model} unavailable."
    assert conf.activation in [
        "tanh",
        "sigmoid",
    ], f"Model activation {conf.activation} unavailable."
    assert conf.loss in ["mse", "hybrid"], f"Model loss {conf.loss} unavailable."


def get_model_config(model_name, backbone, num_labels, id2label):
    if model_name == "maskformer":
        return MaskFormerConfig.from_pretrained(
            backbone,
            num_labels=num_labels,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    elif model_name == "detr":
        return DetrConfig.from_pretrained(
            backbone,
            num_labels=num_labels,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    elif model_name == "beit":
        return BeitConfig.from_pretrained(
            backbone,
            num_labels=num_labels,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        return None
