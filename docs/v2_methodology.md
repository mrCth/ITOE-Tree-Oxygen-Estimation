# SegFormer-B0 (v2) — Training Methodology for IEEE Paper

## 1. Problem Statement
Binary semantic segmentation of tree crowns from high-resolution aerial RGB imagery.
Output per-pixel map {0: background, 1: tree} used for crown delineation, downstream
feature extraction (crown area, count) and carbon/oxygen estimation via allometric equations.

## 2. Dataset
Two-domain training corpus (256×256 tiles), split in `dataset/dataset_split.json`:

| Split | # Images | Source |
|-------|---------|--------|
| Train | 16,932 | NEON RGB + ReforesTree (multi-site aerial) |
| Val   | 3,720  | NEON RGB + ReforesTree |
| Test  | 3,030  | NEON RGB (held-out tiles, same sites) |

Class distribution (Stage 1, pixel level): background 51.4%, tree 48.6%.
ReforesTree subset (Stage 2): train=12,954, val=770; tree class 59.5%.
Test split is NEON-origin; ReforesTree is the fine-tune/validation domain of interest.

## 3. Model Architecture
- **SegFormer-B0** (`nvidia/segformer-b0-finetuned-ade-512-512`) with head replaced,
  `num_labels=2`, `ignore_mismatched_sizes=True`.
- Key design decision: **native 256×256 input resolution** (processor created at
  256×256) instead of the pretrained default 512×512. Rationale: dataset tiles are
  already 256×256; v1's 256→512→256 resize round-trip added bilinear smoothing that
  blurred crown/background boundaries and lost fine crown detail.
- ~3.7M params, efficient for large aerial scenes.

## 4. Training Pipeline (Two-Stage)
Stage 1 — generic segmentation pretraining on ALL data:
- 100 epochs, LR = 6e-5, cosine schedule, warmup = 500 steps
- Effective batch = 8 × 4 (gradient accumulation) = 32
- Weight decay 0.01, FP16 mixed precision, RTX 3060 12GB

Stage 2 — domain adaptation to ReforesTree (high-quality annotations):
- Fine-tune on ReforesTree-only pairs (train=12,954), 20 epochs, LR = 1e-5, no warmup
- Cosine decay, same batch/FP16 settings

## 5. Loss Function
Weighted cross-entropy. Class weights computed from training pixel frequency:
`weight_c = 1 / freq_c`, normalized so sum = 2.
- Stage 1 weights: bg=0.973, tree=1.027
- Stage 2 (ReforesTree) weights: bg=1.191, tree=0.809

## 6. Data Augmentation (Albumentations)
| Transform | Params | Prob |
|-----------|--------|------|
| HorizontalFlip | — | 0.5 |
| VerticalFlip | — | 0.5 |
| RandomRotate90 | — | 0.5 |
| Affine | translate ±5%, scale 0.9–1.1, rotate ±20°, BORDER_REFLECT | 0.5 |
| RandomBrightnessContrast | ±0.2 | 0.4 |
| HueSaturationValue | hue 10, sat 20, val 15 | 0.3 |
| CLAHE | clip 3.0, grid 8×8 | 0.3 |
| GaussNoise | std 0.02–0.1 | 0.3 |
| GaussianBlur | 3×3 | 0.2 |
| CoarseDropout | 2–8 holes, 8–16 px | 0.2 |

Rationale: orientation/position invariance (flip/rotate/affine) for arbitrary crown
angles; spectral robustness (brightness/HSV/CLAHE) across sites and illumination;
sensor-noise robustness (GaussNoise, blur); global-context regularization
(CoarseDropout) against over-segmentation.

## 7. Evaluation Protocol
- Metrics computed **micro-averaged over all pixels** of the val/test set
  (pooled confusion matrix, matching the training-time metric):
  tree IoU, mean IoU, pixel accuracy, precision, recall, F1 (tree class only).
- Early stopping on tree IoU, patience = 8 evaluation rounds, min delta = 1e-4.
- Eval every 2,000 steps; checkpoint every 4,000 steps; seed 42.
- NOTE: `load_best_model_at_end=False` → saved model is the LAST step,
  slightly below the recorded best.

## 8. Results
Validation (ReforesTree val, 770 images), best during training:
- **tree IoU = 0.7308, mean IoU = 0.6553, pixel acc = 0.8020, F1 = 0.8412**

Independent re-evaluation (our scripts, micro / mean-per-image):

ReforesTree val (770):
| Precision | Recall | F1 | micro IoU |
|:---------:|:------:|:--:|:---------:|
| 0.8376 | 0.8433 | 0.8405 | 0.7248 |

NEON test (3,030 images):
| Precision | Recall | F1 | micro IoU |
|:---------:|:------:|:--:|:---------:|
| 0.8570 | 0.8550 | 0.8560 | 0.7482 |

Comparison to other configs (same micro convention, ReforesTree val):
- v1 (NEON-only, 512px pipeline): F1=0.8460, IoU=0.7331, recall 0.9165
- v3 (Dice+CE hybrid loss, stronger augmentation, 100-ep fine-tune): F1=0.8350, IoU=0.7168

## 9. Key Takeaways / Contributions
1. Native 256×256 avoids boundary-smoothing from resolution round-tripping.
2. Two-stage transfer: broad NEON pretraining → ReforesTree domain fine-tune.
3. Weighted CE handles class imbalance (no Dice loss needed).
4. Balanced precision/recall (both ~0.84–0.86) → reliable crown counts/areas for
   downstream allometric carbon & oxygen estimation.

## 10. Reproducibility
- Code: `phase2_train_segformer_v2.py`
- Config: `config.py`
- Model checkpoints: `models/segformer/stage1_v2/`, `stage2_v2/`
- Training log: `train_v2.log` (GPU: NVIDIA RTX 3060 12GB, fp16, seed 42)
- Evaluation scripts: `evaluate_segformer.py`, `evaluate_all_models.py`,
  `evaluate_reforestree_val.py`
