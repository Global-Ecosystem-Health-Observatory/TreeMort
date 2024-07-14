import os
import torch

import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from treeseg.utils.checkpoints import get_checkpoint
#from treeseg.modeling.network.kokonet import Kokonet
#from treeseg.modeling.network.kokonet_hrnet import Kokonet_hrnet
from treeseg.utils.loss import hybrid_loss, mse_loss, iou_score, f_score

def resume_or_load(conf):
    model, optimizer, criterion, metrics = build_model(conf.model, conf.input_channels, conf.output_channels, conf.activation, conf.loss, conf.learning_rate, conf.threshold)

    if conf.resume:
        checkpoint = get_checkpoint(conf.model_weights, conf.output_dir)

        if checkpoint:
            model.load_weights(checkpoint, skip_mismatch=True)
            print(f"Loaded weights from {checkpoint}")

        else:
            print("No checkpoint found. Proceeding without loading weights.")

    else:
        print("Model training from scratch.")

    return model, optimizer, criterion, metrics

def build_model(model_name, input_channels, output_channels, activation, loss, learning_rate, threshold):
    assert model_name in ["unet", "kokonet", "kokonet_hrnet"], f"Model {model_name} unavailable."
    assert activation in ["tanh", "sigmoid"], f"Model activation {activation} unavailable."
    assert loss in ["mse", "hybrid"], f"Model loss {loss} unavailable."

    if model_name == "unet":
        model = smp.Unet(encoder_name="resnet34", in_channels=input_channels, classes=output_channels, activation=None)
    
    elif model_name == "kokonet":
        #model = Kokonet(input_channels=input_channels, output_channels=output_channels, activation=activation)
        pass
    
    elif model_name == "kokonet_hrnet":
        #model = Kokonet_hrnet(input_channels=input_channels, output_channels=output_channels, activation=activation)
        pass

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if loss == "hybrid":
        criterion = hybrid_loss
        def metrics(pred, target):
            return {'iou_score': iou_score(pred, target, threshold), 'f_score': f_score(pred, target, threshold)}
        
    elif loss == "mse":
        criterion = mse_loss
        def metrics(pred, target):
            mse = mse_loss(pred, target)
            mae = nn.functional.l1_loss(pred, target)
            rmse = torch.sqrt(mse)
            return {'mse': mse, 'mae': mae, 'rmse': rmse}

    return model, optimizer, criterion, metrics
