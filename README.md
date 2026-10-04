# Pedestrian Attribute Recognition (UPAR2027)

A PyTorch project for multi-label pedestrian attribute recognition (PAR). Given
an image of a person, the model predicts multiple visual attributes, such as
age-related appearance, clothing, and accessories. The project supports
Market1501, PA100k, and PETA annotations, with a DINOv3-based image model and
an optional prompted region-of-interest (ROI) mixture-of-experts model.

## Overview

Pedestrian images contain evidence at different scales. Some attributes depend
on a broad view of the person, while others may be expressed by a small region.
Attributes can also provide useful context for one another. These properties
make PAR more than a single whole-image classification problem.

The repository provides data loading and augmentation, model training,
multi-label evaluation, checkpointing, and optional Weights & Biases and
Hugging Face Hub integration.

## Problem Statement

The project focuses on three challenges in pedestrian attribute recognition:

1. **Dataset and person variation.** Similar image color and quality statistics
   do not eliminate domain shift. Datasets contain different people, clothing,
   poses, and capture conditions; their image resolutions also vary. Variation
   in the people themselves can shift the visual distribution even when coarse
   RGB/HSV and quality statistics look similar.
2. **Competing evidence and small regions.** PAR predicts many labels from the
   same image. A pixel or region may provide evidence for multiple attributes,
   while large regions such as clothing can dominate a whole-image
   representation. A small but informative region—such as hair, a face, or an
   accessory—can consequently have less influence than its importance for a
   particular label warrants.
3. **Relationships between attributes.** Attributes are not independent:
   context useful for one prediction can help interpret another. Age-related
   appearance, for example, may be judged using both facial and clothing cues.
   A model should be able to use local evidence without losing the broader
   context and dependencies among its predictions.

## Approach

The model designs use complementary global, local, and label-aware
representations. The ROI-MoE design is specifically motivated by challenges 2
and 3:

- **Consistent image preparation:** images are resized to the configured input
  dimensions and normalized with ImageNet statistics. This gives the model a
  common tensor size across datasets, but does not remove semantic domain shift
  between people or datasets.
- **Pretrained visual features:** the default image backbone uses pretrained
  DINOv3 features. The image model uses spatial patch tokens so predictions
  have access to both global context and localized visual information.
- **Local ROI processing:** the ROI-MoE architecture can use YOLO-E or SAM 3
  text prompts to propose crops. Processing a crop as its own input is
  intended to give small or easily overshadowed cues a more direct path into
  the representation, instead of relying only on their contribution to a
  whole-image feature.
- **Learned expert specialization:** sparse top-k routing sends each valid ROI
  to selected experts. The experts are learnable feature processors, not
  experts assigned to particular labels; the gating network learns which
  experts process each region.
- **Global context alongside ROIs:** full-image features are concatenated with
  attended ROI-expert features before classification. This lets the classifier
  use broad person context together with local evidence, including when a
  proposal is incomplete or unhelpful. It is a complementary path, not a
  guarantee that proposal errors will be corrected.
- **Attribute-aware representations and interaction:** in the `image`
  architecture, a learned query for each attribute attends over visual tokens,
  then a transformer processes the set of attribute features. This explicitly
  models relationships among attribute representations. In `roi_moe`,
  attention summarizes ROI-expert features and sparse classification experts
  predict the attribute vector from those features and the full-image
  context.
- **Imbalance-aware training:** focal loss can derive per-attribute positive
  and class weights from the selected training labels.

These components provide modeling strategies for the identified challenges;
they do not constitute a formal domain-adaptation algorithm.

## Model Architectures

### Attribute-aware image model

The default `image` architecture uses a pretrained DINOv3 encoder. Its global
and spatial patch features are presented to learned per-attribute queries
through cross-attention. A two-layer transformer then models interactions
between attribute features, and a shared classifier head outputs one logit per
attribute.

Use `--backbone conv` for a convolutional encoder trained from scratch. Use
`--finetune-backbone` to fine-tune DINOv3; otherwise, its pretrained model
weights are frozen.

### Prompted ROI mixture of experts

The `roi_moe` architecture is motivated by the risk that evidence from small
regions will be diluted by larger regions when the whole image is represented
as one feature. Prompted person regions are cropped and routed to their top-k
ROI experts, which learn to process local visual patterns. Learned attention
queries combine the resulting expert features, and the model concatenates this
local representation with a full-image feature. The global branch preserves
broader context alongside the ROI evidence. A sparse top-k classification
expert layer then predicts the full attribute vector, returning logits shaped
`[batch_size, num_attributes]`.

The ROI experts are not assigned one per label, and the attention queries pool
ROI-expert features rather than directly representing individual labels. The
explicit per-attribute query and label-interaction transformer described
above belong to the `image` architecture. In `roi_moe`, attribute predictions
share the fused representation and selected classification experts. Expert
routing provides a way for the model to learn specialized processing; it is
not a claim that experts are inherently superior to a standard feed-forward
classifier.

YOLO-E and SAM 3 are optional proposal generators. Each returns one
highest-confidence pixel-coordinate `(x1, y1, x2, y2)` box per prompt and image;
missing detections use zero-area boxes and are ignored by the ROI encoder.
Automatic proposal generation can be disabled with `--roi-generator none`, or
caller-provided `roi_boxes` can be used instead.

The ROI model accepts two normalized image tensors plus an optional detector
view:

```python
from src.models import SparseROIAttributeModel

model = SparseROIAttributeModel(
    num_classes=40,
    hidden_dim=128,
    num_experts=5,
    attention_k=2,
    roi_generator="yoloe",
    prompts=("person", "backpack"),
)

# The first two inputs use the configured ImageNet normalization.
# The detector view is resized RGB in [0, 1], before that normalization.
logits = model(
    images_aug,
    images_no_aug,
    images_detector=images_detector,
)
```

The training data loader creates the detector view only when automatic ROI
generation is enabled. It is a separate resized RGB tensor in `[0, 1]`, so
YOLO-E receives the pixel range it expects without changing the normalized
images used by the model. It is not required when automatic proposals are
disabled or when `roi_boxes` are passed directly.

YOLO-E defaults to `yoloe-11s-seg.pt`. Its weights and text encoder are
downloaded on first use. SAM 3 loads the Hugging Face `facebook/sam3` model and
processor at construction time; access may require accepting the model terms
and authenticating with Hugging Face. SAM 3 runs once per image and prompt,
which can increase proposal-generation time.

## Data

Place the annotations and images under `data/`:

```text
data/
├── annotations/
│   ├── train.csv
│   └── val.csv
├── Market1501/
├── PA100k/
└── PETA/
```

Each CSV must contain a `# image` column followed by one binary column per
attribute. Image paths are resolved relative to the repository's `data/`
directory.

```csv
# image,Age-Young,Age-Adult,...
Market1501/bounding_box_train/0002_c1s1_000451_03.jpg,0,1,...
```

Use `--dataset market`, `--dataset pa`, or `--dataset peta` to select a dataset
subset using the configured CSV row ranges. By default, training and validation
use all rows in their respective CSV files; sample-limit options are available
for smaller experiments.

## Requirements and Setup

- Python 3.10 or newer
- Dependencies listed in [requirements.txt](requirements.txt)
- A CUDA-capable GPU is recommended for faster training; CPU execution is also
  supported

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Pretrained DINOv3, YOLO-E, and SAM 3 weights may require network access on
first use. YOLO-E and SAM 3 are only needed for ROI training with those
generators.

### Optional credentials

Create a `.env` file in the repository root when using Weights & Biases or
Hugging Face features:

```env
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token
```

Training without remote logging or Hub upload does not require these values.

## Training

Train the default DINOv3 attribute-aware image model:

```bash
python scripts/train.py \
  --train-csv data/annotations/train.csv \
  --val-csv data/annotations/val.csv \
  --architecture image \
  --backbone dinov3 \
  --batch-size 128 \
  --epochs 5 \
  --lr 0.002 \
  --weight-decay 1e-5 \
  --optimizer adamw \
  --loss focal \
  --output-dir outputs/train
```

Train the ROI mixture-of-experts model with YOLO-E proposals:

```bash
python scripts/train.py \
  --train-csv data/annotations/train.csv \
  --val-csv data/annotations/val.csv \
  --architecture roi_moe \
  --backbone dinov3 \
  --roi-generator yoloe \
  --batch-size 128 \
  --epochs 5 \
  --lr 0.002 \
  --weight-decay 1e-5 \
  --optimizer adamw \
  --loss focal \
  --output-dir outputs/roi_moe
```

The ROI hyperparameters are configurable:

```bash
--hidden-dim 128
--num-experts 5
--roi-top-k 1
--attention-k 2
--segmentation-top-k 3
--num-attn-heads 4
```

`--hidden` aliases `--hidden-dim`; `--num-attention-heads` aliases
`--num-attn-heads`. The default hidden dimensions are 256 for `image` and 128
for `roi_moe`. Run `python scripts/train.py --help` to see all options.

Other useful options:

```bash
--dataset market
--train-num-samples 2000
--eval-num-samples 500
--no-augment
--finetune-backbone
--resume outputs/train/last_epoch.pt
--wandb
--wandb-project UPAR2027
--wandb-run-name experiment-name
--hub-repo-id your-org/your-model-name
--hub-private
```

Training augmentations and ImageNet normalization are configured in
[configs/augmentation.yaml](configs/augmentation.yaml). The input size can be
changed with `--height` and `--width`.

## Training Design

The default settings are read from [configs/train.yaml](configs/train.yaml)
and can be overridden on the command line. Current defaults include five
epochs, batch size 128, learning rate `0.002`, AdamW, focal loss, and `256 x
256` input images.

Supported losses are:

- `bce`: binary cross-entropy with logits
- `weighted_bce`: binary cross-entropy with configured positive weights
- `focal`: focal loss with positive/class weights derived from the training
  labels when available

When DINOv3 is trainable, its optimizer parameter group uses a learning rate
one hundredth of the head learning rate. The optimizer options are `adamw` and
`sgd`.

## Evaluation and Outputs

Validation converts logits to probabilities with sigmoid and uses a default
threshold of `0.5`. Reported metrics include:

- Challenge average
- Label mean accuracy (`mA`)
- Label F1
- Instance accuracy, precision, recall, and F1

Training writes checkpoints and epoch metrics to `outputs/train/` by default:

- `best_mA.pt`
- `best_f1.pt`
- `last_epoch.pt`
- `train_results.csv`

The checkpoints include model, optimizer, scheduler, epoch, and metric state
for resuming training.

## Repository Structure

```text
.
├── configs/       # Training, augmentation, evaluation, prompt, and logging configuration
├── data/          # Annotation CSVs and pedestrian image datasets
├── scripts/       # Training entry point
├── src/
│   ├── builders/  # Data-loader, loss, and optimizer builders
│   ├── metrics/   # Label-, instance-, and challenge-level evaluation
│   ├── models/    # Image and ROI mixture-of-experts architectures
│   ├── train/     # Dataset transforms, training loop, and checkpoints
│   └── utils/     # Configuration, logging, and integration helpers
├── tests/         # Regression and unit tests
├── submission/    # Submission package scaffold
├── requirements.txt
└── README.md
```

## License

This project is distributed under the terms in [LICENSE](LICENSE).
