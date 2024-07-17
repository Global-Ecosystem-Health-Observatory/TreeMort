# Tree Mortality Segmentation

## Getting Started (Puhti)

- Clone repository

```bash
git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeSeg.git
cd TreeSeg
```

- Install packages

```bash
sh scripts/install_treemortality.sh
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
| Kokonet                           | 0.791           | 0.810               |
| Kokonet Binarized                 | 0.796           | 0.818               |
| Kokonet Binarized with Backbone   | 0.759           | 0.781               |
| Unet with Self Attention          | 0.844           | 0.862               |


## (Optional) Dataset Creation

- Create the Train and Test datsets

```bash
python dataset/creator.py /Users/anisr/Documents/AerialImages_ITD
```

- Create Train and Test dataset with a few samples, for testing purposes only

```bash
python dataset/creator.py /Users/anisr/Documents/AerialImages_ITD --no-of-samples 2 
```

- Copy Train and Test folders to Puhti scratch folder

```bash
scp -O -r ~/Documents/AerialImageModel_ITD/Train rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/AerialImageModel_ITD/Train

scp -O -r ~/Documents/AerialImageModel_ITD/Test rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/AerialImageModel_ITD/Test
```

## Getting Started (Mac OS, for development)

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
