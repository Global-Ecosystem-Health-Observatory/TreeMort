import torch
import torch.nn.functional as F


def process_model_output(model, images, model_name):
    _, _, h, w = images.shape

    if model_name == "maskformer":
        outputs = model(images)
        query_logits = outputs['masks_queries_logits']
        combined_logits = torch.max(query_logits, dim=1).values
        interpolated_logits = F.interpolate(combined_logits.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
        logits = interpolated_logits
        features = None

    elif model_name == "detr":
        outputs = model(images)
        query_logits = outputs['pred_masks']
        combined_logits = torch.max(query_logits, dim=1).values
        interpolated_logits = F.interpolate(combined_logits.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
        logits = interpolated_logits
        features = None

    elif model_name in ["dinov2", "beit"]:
        outputs = model(images)
        logits = outputs.logits[:, 1:2, :, :]
        features = None

    elif model_name in ("flair_unet", "flair_unet_dann"):
        logits, features = model(images)

    else:
        logits = model(images)
        features = None

    return logits, features


def prepare_pred_and_target(logits, labels, target_hw):
    _, _, h, w = labels.shape
    th, tw = target_hw

    # Buffer is always channel 3 in labels
    buffer_mask = labels[:, 3:4, :, :]
    buffer_mask = center_crop(buffer_mask, (th, tw))   # [B,1,h,w]

    # Crop logits channel by channel (works for C=1 or C=3)
    C = logits.shape[1]
    cropped_logits = torch.cat(
        [center_crop(logits[:, i:i+1, :, :], (th, tw)) for i in range(C)],
        dim=1
    )  # [B,C,h,w]

    # Build matched targets from the first C semantic channels of labels
    # label channels: 0=mask, 1=centroid, 2=hybrid
    tgt_parts = []
    for i in range(min(C, 3)):
        tgt_parts.append(center_crop(labels[:, i:i+1, :, :], (th, tw)))
    targets = torch.cat(tgt_parts, dim=1) if tgt_parts else None  # [B,C,h,w]

    return cropped_logits, targets, buffer_mask


def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return tensor[..., i:i+th, j:j+tw]