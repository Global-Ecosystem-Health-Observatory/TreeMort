import torch
import torch.nn.functional as F

from transformers import AutoImageProcessor


def get_image_processor(model_name, backbone):
    if model_name in ["maskformer", "detr", "beit", "dinov2"]:
        image_processor = AutoImageProcessor.from_pretrained(backbone)

        if model_name == "beit":
            image_processor.size["shortest_edge"] = min(image_processor.size["height"], image_processor.size["width"])
            image_processor.do_pad = False
        elif model_name in ["maskformer", "dinov2"]:
            image_processor.do_pad = False
    else:
        image_processor = None


def apply_image_processor(image, label, image_processor):
    image = _rescale_image(image, image_processor) if image_processor.do_rescale else image
    image, label = _resize_image_and_label(image, label, image_processor) if image_processor.do_resize else (image, label)
    image = _normalize_image(image, image_processor) if image_processor.do_normalize else image
    image, label = _pad_image_and_label(image, label, image_processor) if image_processor.do_pad else (image, label)
    return image, label


def _rescale_image(image, image_processor):
    return image * image_processor.rescale_factor


def _resize_image_and_label(image, label, image_processor):
    new_size = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    
    image = F.interpolate(image.unsqueeze(0), size=new_size, mode="bilinear", align_corners=False).squeeze(0)
    label = F.interpolate(label.unsqueeze(0), size=new_size, mode="nearest").squeeze(0)

    return image, label


def _normalize_image(image, image_processor):
    image_mean = torch.tensor(image_processor.image_mean)
    image_std = torch.tensor(image_processor.image_std)

    return (image - image_mean.view(-1, 1, 1)) / image_std.view(-1, 1, 1)


def _pad_image_and_label(image, label, image_processor):
    pad_size = getattr(image_processor, "pad_size", None)

    if pad_size:
        pad_h, pad_w = pad_size[0] - image.shape[1], pad_size[1] - image.shape[2]

        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
            label = F.pad(label, (0, pad_w, 0, pad_h), value=0)
    
    return image, label
