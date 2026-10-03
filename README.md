# Pedestrian Attribute Recognition (UPAR2027)

A PyTorch-based pedestrian attribute recognition pipeline for multi-label classification of human attributes such as age, gender, clothing color, accessories, and garment style. The project is built around a ResNet backbone and supports configurable training, optional augmentation, checkpointing, Weights & Biases logging, and Hugging Face uploads.

## Overview

This repository trains a multi-label classifier on CSV-annotated pedestrian images. Each sample is represented as an image path plus binary attribute labels. The training code builds a classification head on top of a pretrained torchvision ResNet model and optimizes it with a binary cross-entropy style loss function.

The project includes:

- ResNet18 / ResNet50 backbones from torchvision
- Multi-label attribute prediction using sigmoid outputs
- Training and evaluation CSV support
- Dataset subset selection for Market1501, PA100k, and PETA
- Data augmentation switches for training
- Checkpointing and metric logging
- Optional W&B and Hugging Face Hub integration

## Repository structure

```text
.
├── configs/
│   ├── augmentation.yaml
│   ├── eval.yaml
│   ├── miscs.yaml
│   └── train.yaml
├── data/
│   ├── annotations/
│   ├── Market1501/
│   ├── PA100k/
│   └── PETA/
├── outputs/
│   └── train/
├── scripts/
│   └── train.py
├── src/
│   ├── builders/
│   ├── losses/
│   ├── metrics/
│   ├── models/
│   ├── train/
│   └── utils/
├── tests/
├── .env
├── LICENSE
├── requirements.txt
├── README.md
└── submission/
```

## Requirements

- Python 3.10+
- PyTorch and Torchvision
- Optional: CUDA-capable GPU for faster training

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment configuration

The project reads environment variables from a `.env` file using `python-dotenv`.

Create a `.env` file in the project root with the following variables when using W&B or Hugging Face upload features:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

If you do not need remote logging or Hub uploads, the training pipeline can still run without these values.

## Dataset setup

The expected data layout is:

```text
data/
├── annotations/
│   ├── train.csv
│   └── val.csv
├── Market1501/
├── PA100k/
├── PETA/
```

The CSV files are expected to have a `# image` column followed by binary attribute columns. The loader resolves each image path relative to the repository root by prepending `data/`.

Example:

```csv
# image,Age-Young,Age-Adult,...
Market1501/bounding_box_train/0002_c1s1_000451_03.jpg,0,1,0,...
```

The project also supports dataset subset selection with the `--dataset` flag using values:

- `market`
- `pa`
- `peta`

## Training

Run training from the project root:

```bash
python scripts/train.py \
  --train-csv data/annotations/train.csv \
  --val-csv data/annotations/val.csv \
  --backbone resnet50 \
  --pretrained \
  --batch-size 128 \
  --epochs 5 \
  --lr 3e-4 \
  --weight-decay 1e-5 \
  --optimizer adamw \
  --loss bce \
  --output-dir outputs/train
```

Common training options:

```bash
python scripts/train.py --help
```

Some useful flags include:

```bash
--dataset market
--all-training-data
--augment
--wandb
--wandb-project UPAR2027
--wandb-entity denison
--hub-repo-id your-org/your-model-name
--resume outputs/train/last_epoch.pt
```

## Configuration

Training behavior is controlled by YAML files under `configs/`:

- `configs/train.yaml`: learning rate, epochs, optimizer, batch size, loss, output directory, augmentation flags
- `configs/eval.yaml`: validation sampling settings
- `configs/miscs.yaml`: W&B project/entity defaults

The default training config currently sets:

```yaml
default:
  seed: 42

hyper_param:
  lr: 3.e-4
  epochs: 5
  batch_size: 128
  loss: "bce"
  optim: "adamw"
```

## Outputs

Training writes checkpoints and metrics to `outputs/train/` by default. The checkpoint names include:

- `best_mA.pt`
- `best_f1.pt`
- `last_epoch.pt`
- `train_results.csv`

The CSV stores epoch-wise metrics including:

- `epoch`
- `train_loss`
- `eval_loss`
- `challenge_avg`
- `mA`
- `label_f1`
- `inst_acc`
- `inst_prec`
- `inst_rec`
- `inst_f1`
- `learning_rate`

## Evaluation metrics

Validation metrics are computed with a sigmoid threshold of 0.5. The code reports:

- Challenge average
- Label mean accuracy (`mA`)
- Label F1
- Instance accuracy, precision, recall, and F1

This follows the multi-label classification evaluation flow in `src/metrics/run.py`.

## Sparse ROI mixture-of-experts model

`src.models.SparseROIAttributeModel` provides a separate model implementation for
prompted pedestrian ROIs and full-image context. It uses sparse top-k routing to
send each ROI to only its selected ROI expert, attends over the per-expert
features with `attention_k` learned queries, concatenates that feature with the
full-image feature, and routes the fused representation through a five-expert
classification head. The returned tensor contains **logits** shaped
`[batch_size, num_classes]`; train it with a multi-label logits loss such as
`BCEWithLogitsLoss`.

```python
from src.models import SparseROIAttributeModel

model = SparseROIAttributeModel(
    num_classes=40,
    hidden_dim=256,
    num_experts=4,
    attention_k=2,
    prompts=("person", "backpack"),
)
logits = model(images, roi_boxes=boxes)
```

Boxes use pixel-coordinate `(x1, y1, x2, y2)` format with shape `[B, R, 4]`;
zero-area boxes can be used for padding. To connect a detector such as DINO or
SAM 3, pass a `proposal_generator(images, prompts)` callable that returns this
box tensor. Detector loading and prompt-specific post-processing remain the
caller's responsibility. The default image/ROI encoders are compact convolutional
encoders; custom encoders may be passed in if they return `[batch, hidden_dim]`.

## Notes

- The code uses `argparse` for command-line configuration; no separate training notebook is required.
- Augmentation is controlled by flags in the config and can be toggled with the `--augment` option.
- The project includes optional Weights & Biases logging and Hugging Face folder uploads for trained artifacts.

## License

This project is distributed under the MIT license. See [LICENSE](LICENSE) for details.
