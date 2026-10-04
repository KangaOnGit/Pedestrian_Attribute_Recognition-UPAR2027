# Pedestrian Attribute Recognition (UPAR2027)

A PyTorch-based pedestrian attribute recognition pipeline for multi-label classification of human attributes such as age, gender, clothing color, accessories, and garment style. The default training model uses pretrained DINOv3 global and regional image features with attribute-aware attention. A prompted ROI mixture-of-experts model is available as an alternate architecture.

## Overview

This repository trains a multi-label classifier on CSV-annotated pedestrian images. Each sample is represented as an image path plus binary attribute labels. Training supports a pretrained DINOv3 encoder, an image-only convolutional baseline, and an optional prompted ROI mixture-of-experts model.

The project includes:

- Pretrained DINOv3 global and spatial-region features
- Attribute-query cross-attention and label-interaction layers
- Optional prompted ROI mixture-of-experts architecture
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
  --architecture image \
  --backbone dinov3 \
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
--train-num-samples 2000
--eval-num-samples 500
--augment
--wandb
--wandb-project UPAR2027
--wandb-entity denison
--hub-repo-id your-org/your-model-name
--resume outputs/train/last_epoch.pt
```

Training and validation use their full CSV splits by default. The sample-count
flags above are optional limits for faster experiments.

## Configuration

Training behavior is controlled by YAML files under `configs/`:

- `configs/train.yaml`: learning rate, epochs, optimizer, batch size, loss, output directory, augmentation flags
- `configs/eval.yaml`: validation sampling settings
- `configs/miscs.yaml`: W&B project/entity defaults

When using focal loss, per-attribute positive and class weights are calculated
from the labels in the selected training set to account for label imbalance.

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

## Model architectures

The default `image` architecture uses a pretrained DINOv3 image encoder. Its
global feature and individual spatial patch features are presented to learned
per-attribute queries. Those queries attend over visual tokens, then interact
through a transformer encoder before producing attribute logits. This preserves
global context while allowing each attribute to emphasize relevant image
patches and model dependencies with other attributes.

Use `--backbone conv` for a from-scratch baseline, or `--finetune-backbone` to
fine-tune DINOv3 with a lower learning rate than the classification head.

`--architecture roi_moe` selects `src.models.SparseROIAttributeModel` for
prompted pedestrian ROIs and full-image context. It uses sparse top-k routing to
send each ROI to only its selected ROI expert, attends over the per-expert
features with `attention_k` learned queries, concatenates that feature with the
full-image feature, and routes the fused representation through a five-expert
classification head. The returned tensor contains **logits** shaped
`[batch_size, num_classes]`; train it with a multi-label logits loss such as
`BCEWithLogitsLoss`.

The ROI model parameters can be set on the training command:

```bash
python scripts/train.py \
  --architecture roi_moe \
  --hidden-dim 128 \
  --num-experts 5 \
  --roi-top-k 1 \
  --attention-k 2 \
  --segmentation-top-k 3 \
  --num-attn-heads 4
```

`--hidden` is also accepted as an alias for `--hidden-dim`, and
`--num-attention-heads` is an alias for `--num-attn-heads`. Hidden dimension
defaults remain 256 for the `image` architecture and 128 for `roi_moe`.

```python
from src.models import SparseROIAttributeModel

model = SparseROIAttributeModel(
    num_classes=40,
    hidden_dim=256,
    num_experts=4,
    attention_k=2,
    roi_generator="yoloe",
    prompts=("person", "backpack"),
)
logits = model(images)
```

YOLO-E is the default prompt-based ROI generator. It applies the text prompts
once and detects all images in each batch together, returning
pixel-coordinate `(x1, y1, x2, y2)` boxes. Its default checkpoint is
`yoloe-11s-seg.pt`; pass `yoloe_model_id` or `yoloe_score_threshold` to customize
it. The first prompted run downloads the YOLO-E weights and its text encoder,
so it requires network access.

Setting `roi_generator="sam3"` loads the Hugging Face `facebook/sam3` model and
processor during model construction; SAM 3 runs each configured text prompt and
returns pixel-coordinate `(x1, y1, x2, y2)` boxes. SAM 3 weights may require
accepting the upstream model terms and authenticating with Hugging Face. Set
`roi_generator="none"` to disable automatic proposals, or pass `roi_boxes`
directly to `forward`; direct boxes take precedence over generated boxes.
SAM 3 remains available as an alternative but runs once per image and prompt,
so larger batches or prompt lists increase proposal-generation time.

DINOv3 is available as an optional full-image feature encoder, not as a
text-prompted box detector:

```python
model = SparseROIAttributeModel(
    num_classes=40,
    hidden_dim=256,
    num_experts=4,
    attention_k=2,
    roi_generator="sam3",
    prompts=("person", "backpack"),
    image_backbone_type="dinov3",
)
```

The default full-image and ROI encoders are compact convolutional encoders.
Images are expected to use the ImageNet mean/std normalization configured by the
training pipeline; the detector and DINOv3 preprocessing convert them back to
RGB before their Hugging Face processors run. Custom image and ROI encoders may
be passed in if they return `[batch, hidden_dim]`.

## Notes

- The code uses `argparse` for command-line configuration; no separate training notebook is required.
- Augmentation is controlled by flags in the config and can be toggled with the `--augment` option.
- The project includes optional Weights & Biases logging and Hugging Face folder uploads for trained artifacts.

## License

This project is distributed under the MIT license. See [LICENSE](LICENSE) for details.
