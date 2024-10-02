import os
import torch
import argparse

from treemort.data.loader import prepare_datasets
from treemort.utils.config import setup
from treemort.modeling.builder import resume_or_load


def run(conf):

    id2label = {0: "alive", 1: "dead"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, _ = prepare_datasets(conf)

    print(len(train_loader))
    model, _, criterion, _, _ = resume_or_load(conf, id2label, len(train_loader), device)

    model.eval()

    fisher_information = {name: torch.zeros(param.size()).to(param.device) for name, param in model.named_parameters() if param.requires_grad}

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_information[name] += (param.grad ** 2) / len(train_loader)

    optimal_parameters = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    ewc_data = {
        "optimal_parameters": {name: param.cpu() for name, param in optimal_parameters.items()},
        "fisher_information": {name: fisher.cpu() for name, fisher in fisher_information.items()}
    }

    torch.save(ewc_data, os.path.join(conf.output_dir, 'ewc_data.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    conf = setup(args.config)

    run(conf, args.eval_only)


'''
Usage:

- Train

python -m treemort.utils.fisher ./configs/flair_unet_bs8_cs256.txt

'''