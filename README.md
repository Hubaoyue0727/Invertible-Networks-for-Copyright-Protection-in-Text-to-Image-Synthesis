# Invertible-Networks-for-Copyright-Protection-in-Text-to-Image-Synthesis
Who Controls the Authorization? Invertible Networks for Copyright Protection in Text-to-Image Synthesis

## Method

## Training Demo
``` python
python train.py \
    --inputpath_all "./data/VGGFace2_demo"\
    --copyrightpath "./data/copyright.png" \
    --T2Imodel "stabilityai/stable-diffusion-2-1"
```

## Training

### Environment Setup
This project provides both `environment.yml` (Conda) and `requirements.txt` (pip).

#### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate ldm_db
```

#### Option 2: pip
```bash
pip install -r requirements.txt
```

### Dataset Preparation
Place your original images and watermark file in the following structure:

data/
│
├── Original/                  # Original images to be protected
└── watermark.png              # Copyright watermark

### Training
```bash
bash scripts/train.sh
```

## BibTeX
@inproceedings{hu2025controls,
  title={Who Controls the Authorization? Invertible Networks for Copyright Protection in Text-to-Image Synthesis},
  author={Hu, Baoyue and Wei, Yang and Xiao, Junhao and Huang, Wendong and Bi, Xiuli and Xiao, Bin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15832--15841},
  year={2025}
}
