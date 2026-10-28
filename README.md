# CFR-Net

Official PyTorch implementation of **CFR-Net**, a cross-modal feature calibration and refinement network for lesion detection in paired CT and PET images.

CFR-Net uses modality-specific feature extraction branches to preserve CT and PET representations. Multi-scale features are aligned and fused by the Two-Stage Adaptive Feature Fusion (TSAFF) module, followed by a feature pyramid network and a YOLOX detection head.

## Architecture

![CFR-Net architecture](assets/figure2_architecture.png)

The TSAFF module contains two stages. The first stage performs spatial feature compression and iterative Q-V-gated cross-modal interaction. The second stage uses fuzzy uncertainty suppression to refine spatial responses and improve lesion localization.

More details are available in [the architecture documentation](docs/architecture.md).

## Repository structure

```text
CFR-PETCT/
├── assets/                    # Architecture figures
├── docs/                      # Method documentation
├── nets/
│   ├── yolo_tsaff.py          # CFR-Net model entry point
│   ├── tsaff_modified.py      # TSAFF, SFM, QVGSIM, IFRM and FUSM
│   ├── darknet.py             # Dual-branch feature extractor
│   ├── yolo.py                # Feature pyramid and YOLOX head
│   └── yolo_training.py       # Detection loss and training utilities
├── utils/                     # Data loading, optimization and evaluation
├── train_tsaff.py             # Training
├── test_tsaff.py              # Evaluation
├── predict.py                 # Inference
├── get_map.py                 # mAP evaluation
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<your-account>/CFR-PETCT.git
cd CFR-PETCT

python -m venv .venv
source .venv/bin/activate        # Linux or macOS
# .venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA version when GPU acceleration is required.

## Dataset preparation

The training pipeline expects paired CT and PET images with Pascal VOC-style XML annotations. Keep clinical data outside version control.

The default directory layout is:

```text
data/
├── classes.txt
├── Annotations/
├── JPEGImages_CT/
├── JPEGImages_PET/
└── ImageSets/
    └── Main/
        ├── 2007_train.txt
        └── 2007_val.txt
```

Paths can be configured through environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `DATA_ROOT` | Dataset directory | `./data` |
| `CLASSES_PATH` | Class-name file | `./data/classes.txt` |
| `MODEL_PATH` | Initial or pretrained checkpoint | `./weights/yolox_l.pth` |
| `TRAIN_LIST` | Training annotation list | `./data/ImageSets/Main/2007_train.txt` |
| `VAL_LIST` | Validation annotation list | `./data/ImageSets/Main/2007_val.txt` |
| `SAVE_DIR` | Checkpoints and logs | `./logs` |

## Model check

Run a forward-pass check before training:

```bash
python check_tsaff.py --phi l --loops 2 --size 640
```

## Training

```bash
python train_tsaff.py
```

Example with custom paths:

```bash
DATA_ROOT=/path/to/dataset \
MODEL_PATH=/path/to/yolox_l.pth \
SAVE_DIR=./logs \
python train_tsaff.py
```

## Evaluation

```bash
python test_tsaff.py \
  --checkpoint ./weights/best_epoch_weights.pth \
  --val-list ./data/ImageSets/Main/2007_val.txt \
  --classes ./data/classes.txt
```

## Inference

```bash
python predict.py
```

Set the checkpoint and class paths in the inference configuration or pass them through the corresponding model arguments.

## Citation

If this code is useful in your research, please cite the associated paper. The BibTeX entry will be added after publication.

## Data availability

Clinical images and annotations are not distributed in this repository. Access to the data is subject to the requirements stated in the associated study.

