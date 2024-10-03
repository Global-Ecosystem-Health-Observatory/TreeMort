import torch
import torch.nn.functional as F


def process_model_output(model, images, conf):
    _, _, h, w = images.shape

    if conf.model == "maskformer":
        outputs = model(images)
        query_logits = outputs['masks_queries_logits']
        combined_logits = torch.max(query_logits, dim=1).values
        interpolated_logits = F.interpolate(combined_logits.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
        logits = interpolated_logits
    
    elif conf.model == "detr":
        outputs = model(images)
        query_logits = outputs['pred_masks']
        combined_logits = torch.max(query_logits, dim=1).values
        interpolated_logits = F.interpolate(combined_logits.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
        logits = interpolated_logits

    elif conf.model in ["dinov2", "beit"]:
        outputs = model(images)
        logits = outputs.logits[:, 1:2, :, :]
    
    else:
        logits = model(images)
    
    return logits
