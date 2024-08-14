import torch


def process_model_output(model, images, conf, image_processor, labels, device):
    target_sizes = [(label.shape[1], label.shape[2]) for label in labels]

    if conf.model == "maskformer":
        outputs = model(images)
        predictions = image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        predictions = torch.stack([prediction.float().unsqueeze(0).to(device) for prediction in predictions], dim=0)
    
    elif conf.model == "detr":
        outputs = model(images)
        predictions = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)
        predictions = torch.stack([prediction['segmentation'].float().unsqueeze(0).to(device) for prediction in predictions], dim=0)
        
    elif conf.model == "beit":
        outputs = model(images)
        predictions = torch.sigmoid(outputs.logits[:, 1:2, :, :]) 

    elif conf.model == "dinov2":
        outputs = model(images)
        predictions = torch.sigmoid(outputs.logits[:, 1:2, :, :]) 

    else:
        predictions = model(images)
    
    return predictions
