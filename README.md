# Tree Mortality Segmentation

## Getting Started (Mac OS)

- Clone repository

```bash
git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeSeg.git
cd TreeSeg
```

- Create a new environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --quiet -r requirements.txt
```

- Train a model

```bash
python train_net.py ./configs/unet_self_attention_bs8_cs256.txt
```

- Evaluate the model
```bash
python train_net.py ./configs/unet_self_attention_bs8_cs256.txt --eval-only
```

## Getting Started (Puhti)

- Clone repository

```bash
git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeSeg.git
cd TreeSeg
```

- Create a new environment

```bash
pip install .
```

- Train a model

```bash
sh ./scripts/run_treemortality.sh ./configs/unet_self_attention_bs8_cs256.txt --eval-only false
```

- Evaluate the model
```bash
sh ./scripts/run_treemortality.sh ./configs/unet_self_attention_bs8_cs256.txt --eval-only true
```

## Results


| Model                             | Mean IOU Pixels   | Mean IOU Trees    |
| :-------------------------------: | :---------------: | :---------------: |
| Kokonet                           |   0.791           |   0.810           |
| Kokonet Binarized                 |   0.796           |   0.818           |
| Kokonet Binarized with Backbone   |   0.759           |   0.781           |
