"""
Phase 2 — SegFormer Training
2-stage training on NEON + ReforesTree.

Stage 1: Pre-train on all data (NEON + ReforesTree)
Stage 2: Fine-tune on ReforesTree

Reads from:
    dataset/dataset_split.json

Writes to:
    models/segformer/stage1_all/
    models/segformer/stage2_ref/
    models/segformer_tree_crown/  (final copy)

Flags:
    --no-class-weights    Skip class weights, use uniform CE loss
    --filter-empty        Remove images with all-zero masks before training
"""

import argparse, gc, json, os, sys, shutil, torch, numpy as np, random, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "train.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

from PIL import Image
from torch.utils.data import Dataset, Subset
from torch.nn import functional as F
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments, Trainer,
)
import evaluate
import albumentations as A

import config

# ============================================================================
# Paths
# ============================================================================
DATASET_SPLIT = os.path.join(config.DATASET_DIR, "dataset_split.json")

# Training schedule constants
EVAL_STEPS = 2000    # Evaluate every N steps (separated from save)
SAVE_STEPS = 4000    # Save checkpoint every N steps
EVAL_ACCUM_STEPS = 4 # Accumulate eval batches to reduce GPU memory spikes

def _get_dirs(suffix=None):
    """Return (stage1_dir, stage2_dir, final_dir) with optional suffix."""
    base = os.path.join(config.MODELS_DIR, "segformer")
    if suffix:
        s1 = os.path.join(base, f"stage1_{suffix}")
        s2 = os.path.join(base, f"stage2_{suffix}")
        f = os.path.join(config.MODELS_DIR, f"segformer_tree_crown_{suffix}")
    else:
        s1 = os.path.join(base, "stage1_all")
        s2 = os.path.join(base, "stage2_ref")
        f = os.path.join(config.MODELS_DIR, "segformer_tree_crown")
    return s1, s2, f

ID2LABEL = {0: "background", 1: "tree"}
LABEL2ID = {"background": 0, "tree": 1}

# ============================================================================
# Augmentations
# ============================================================================
TRAIN_TRANSFORMS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
])

# ============================================================================
# Dataset — reads dataset_split.json directly
# ============================================================================
class DatasetSplitDataset(Dataset):
    """
    Loads image-mask pairs from dataset_split.json.
    Handles all 3 mask formats:
      - neon_rgb:  masks/same_name.png
      - reforestree: masks/name_mask.png
      - Tree-Top-View: v1/train/masks/name.png
    """

    def __init__(self, pairs, processor, augment=False):
        self.pairs = pairs
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        img_path = item["image"]
        mask_path = item["mask"]

        image = Image.open(img_path).convert("RGB")
        mask = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.int64)

        if self.augment:
            aug = TRAIN_TRANSFORMS(image=np.array(image), mask=mask.astype("uint8"))
            image = Image.fromarray(aug["image"])
            mask = aug["mask"].astype(np.int64)

        inputs = self.processor(
            images=image, segmentation_maps=mask, return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


class SubsetDataset(Dataset):
    """Wraps DatasetSplitDataset with index subset for train/val split."""

    def __init__(self, base, indices, augment):
        self.base = base
        self.indices = indices
        self.base.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


# ============================================================================
# Metrics
# ============================================================================
metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        if hasattr(logits, 'predictions'):
            logits = logits.predictions
        if hasattr(labels, 'label_ids'):
            labels = labels.label_ids
        logits = np.asarray(logits, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        upsampled = F.interpolate(
            torch.from_numpy(logits),
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1).numpy().astype(np.int64)

        mask = labels != 255
        pred_flat = upsampled[mask]
        ref_flat = labels[mask]

        ious = []
        for c in range(2):
            p = pred_flat == c
            r = ref_flat == c
            inter = (p & r).sum()
            union = (p | r).sum()
            ious.append(float(inter / (union + 1e-8)))

        return {
            "mean_iou": float(np.mean(ious)),
            "tree_iou": float(ious[1]),
        }


# ============================================================================
# GPU Memory / Logging Callback
# ============================================================================
from transformers import TrainerCallback

class GPULoggerCallback(TrainerCallback):
    """Log GPU memory + training metrics every logging_steps."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        gpu_mem = ""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                peak_alloc = torch.cuda.max_memory_allocated(i) / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_mem += (f"  GPU{i}: alloc={alloc:.2f} res={reserved:.2f}"
                            f" peak_alloc={peak_alloc:.2f} peak_res={peak_reserved:.2f}"
                            f" total={total:.1f} GB")
        step = state.global_step
        epoch = state.epoch
        loss = logs.get("loss", "N/A")
        lr = logs.get("learning_rate", "N/A")
        log.info(
            f"Step {step} | Epoch {epoch:.2f} | Loss {loss} | LR {lr} |{gpu_mem}"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log.info(
                f"Eval @ step {state.global_step}: "
                f"mean_iou={metrics.get('eval_mean_iou', 'N/A'):.4f}  "
                f"tree_iou={metrics.get('eval_tree_iou', 'N/A'):.4f}  "
                f"loss={metrics.get('eval_loss', 'N/A')}"
            )
        # Log peak memory after eval and reset for next interval
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                peak_alloc = torch.cuda.max_memory_allocated(i) / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved(i) / 1024**3
                log.info(f"  GPU{i} peak after eval: alloc={peak_alloc:.2f} res={peak_reserved:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    def on_train_begin(self, args, state, control, **kwargs):
        log.info("=" * 60)
        log.info("TRAINING START")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info(f"  GPU {i}: {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM")
        log.info(f"  Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} grad_accum")
        log.info(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        log.info(f"  FP16: {args.fp16}")
        log.info(f"  Eval accum steps: {args.eval_accumulation_steps}")
        log.info("=" * 60)


# ============================================================================
# Eval Memory Guard — clears CUDA cache before evaluation to prevent OOM
# ============================================================================
class EvalMemoryGuard(TrainerCallback):
    """Free GPU memory before eval to prevent OOM on limited VRAM."""

    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Weighted Trainer
# ============================================================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        self.class_weights = class_weights
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = F.interpolate(
            logits, size=labels.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(upsampled, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Helpers
# ============================================================================
def load_split(split_name):
    """Load pairs from dataset_split.json by split name."""
    with open(DATASET_SPLIT) as f:
        data = json.load(f)
    pairs = data[split_name]
    print(f"  {split_name}: {len(pairs)} pairs")
    return pairs


def compute_class_weights(pairs):
    """Compute class weights from a list of {image, mask, source} dicts."""
    total = np.zeros(2, dtype=np.int64)
    for item in pairs:
        mask = np.array(Image.open(item["mask"]).convert("L"))
        mask_bin = (mask > 0).astype(np.int64)
        for c in range(2):
            total[c] += (mask_bin == c).sum()
    freq = total / total.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * 2
    tree_pct = total[1] / total.sum() * 100
    print(f"  Class distribution: bg={total[0]}, tree={total[1]} ({tree_pct:.1f}%)")
    print(f"  Class weights: bg={weights[0]:.3f}, tree={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def make_split(all_pairs, val_ratio=0.1, seed=42):
    """Split all_pairs into train/val using shuffle + ratio."""
    indices = list(range(len(all_pairs)))
    random.seed(seed)
    random.shuffle(indices)
    n_val = max(1, int(len(all_pairs) * val_ratio))
    return indices[n_val:], indices[:n_val]


def get_latest_checkpoint(output_dir):
    """Find latest checkpoint dir in output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"  Latest checkpoint: {latest}")
    return latest


def run_stage(
    stage_name, train_pairs, val_pairs, processor,
    output_dir, epochs, lr, batch_size, grad_accum, class_weights,
    resume_from=None,
):
    """Run one training stage. Returns path to saved model."""
    os.makedirs(output_dir, exist_ok=True)

    train_ds = DatasetSplitDataset(train_pairs, processor, augment=True)
    val_ds = DatasetSplitDataset(val_pairs, processor, augment=False)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    if resume_from and os.path.isdir(resume_from):
        print(f"  Resuming from checkpoint: {resume_from}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            resume_from, num_labels=2,
            id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=grad_accum,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=5,
        # Eval+save separated to avoid simultaneous GPU memory spikes
        # load_best disabled to avoid extra alloc after eval
        load_best_model_at_end=False,
        metric_for_best_model="tree_iou",
        logging_steps=50,
        remove_unused_columns=False,
        fp16=True,
        # Accumulate eval batches — prevents GPU OOM on limited VRAM
        eval_accumulation_steps=EVAL_ACCUM_STEPS,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none",
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        class_weights=class_weights,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[GPULoggerCallback(), EvalMemoryGuard()],
    )

    print(f"  Starting {stage_name}...")
    trainer.train(resume_from_checkpoint=resume_from)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"  Saved -> {output_dir}")

    return output_dir


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Use uniform CE loss instead of class-balanced weights")
    parser.add_argument("--filter-empty", action="store_true",
                        help="Remove images with all-zero masks before training")
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU device id to use (default: 1)")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Custom suffix for model directory name")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config.EPOCHS (default: config value)")
    args = parser.parse_args()

    if args.suffix:
        suffix = args.suffix
    elif args.no_class_weights and not args.filter_empty:
        suffix = "noweights"
    elif args.filter_empty and not args.no_class_weights:
        suffix = "noempty"
    elif args.filter_empty and args.no_class_weights:
        suffix = "noweights_noempty"
    else:
        suffix = None
    STAGE1_DIR, STAGE2_DIR, FINAL_DIR = _get_dirs(suffix)

    if args.epochs is not None:
        config.EPOCHS = args.epochs
        print(f"Overriding EPOCHS to {config.EPOCHS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} (GPU {args.gpu})  |  Class weights: {not args.no_class_weights}  |  Filter empty: {args.filter_empty}  |  Suffix: {suffix or 'default'}  |  Epochs: {config.EPOCHS}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be slow!")

    # Load processor
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    processor.do_reduce_labels = False

    # Load all data
    print("\n=== Loading dataset_split.json ===")
    all_train = load_split("train")
    all_val = load_split("val")

    # Optional: filter empty masks
    if args.filter_empty:
        print("\n=== Filtering empty masks ===")
        def has_tree(item):
            m = np.array(Image.open(item["mask"]).convert("L"))
            return m.max() > 0
        before_t, before_v = len(all_train), len(all_val)
        all_train = [p for p in all_train if has_tree(p)]
        all_val = [p for p in all_val if has_tree(p)]
        print(f"  Train: {before_t} -> {len(all_train)}")
        print(f"  Val: {before_v} -> {len(all_val)}")

    # Compute class weights (or skip)
    if args.no_class_weights:
        class_weights = None
        print("\n=== Class weights disabled (uniform CE loss) ===")
    else:
        print("\n=== Computing class weights ===")
        class_weights = compute_class_weights(all_train)

    # =========================================================================
    # Stage 1: ALL DATA
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"STAGE 1: ALL DATA ({config.EPOCHS} epochs)")
    print("=" * 60)

    s1_resume = get_latest_checkpoint(STAGE1_DIR)
    if not s1_resume:
        check_final = os.path.join(FINAL_DIR, "config.json")
        if os.path.isfile(check_final):
            print(f"  Final model exists ({check_final}), skipping Stage 1")
            stage1_model = FINAL_DIR
        else:
            stage1_model = run_stage(
                stage_name="Stage 1: ALL",
                train_pairs=all_train,
                val_pairs=all_val,
                processor=processor,
                output_dir=STAGE1_DIR,
                epochs=config.EPOCHS,
                lr=config.LEARNING_RATE,
                batch_size=config.BATCH_SIZE,
                grad_accum=config.GRAD_ACCUM_STEPS,
                class_weights=class_weights,
            )
    else:
        stage1_model = run_stage(
            stage_name="Stage 1: ALL",
            train_pairs=all_train,
            val_pairs=all_val,
            processor=processor,
            output_dir=STAGE1_DIR,
            epochs=config.EPOCHS,
            lr=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            grad_accum=config.GRAD_ACCUM_STEPS,
            class_weights=class_weights,
            resume_from=s1_resume,
        )

    # =========================================================================
    # Stage 2: REFORESTREE (fallback: NEON all data if no ReforesTree)
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"STAGE 2: REFORESTREE ({config.FINETUNE_EPOCHS} epochs)")
    print("=" * 60)

    ref_train = [p for p in all_train if p["source"].startswith("reforestree")]
    ref_val = [p for p in all_val if p["source"].startswith("reforestree")]

    if len(ref_train) > 0:
        print(f"  ReforesTree data found: train={len(ref_train)}, val={len(ref_val)}")
        if len(ref_val) < 2:
            ref_train_idx, ref_val_idx = make_split(ref_train, val_ratio=0.2)
            ref_train_pairs = [ref_train[i] for i in ref_train_idx]
            ref_val_pairs = [ref_train[i] for i in ref_val_idx]
        else:
            ref_train_pairs = ref_train
            ref_val_pairs = ref_val
    else:
        print("  WARNING: No ReforesTree data — using ALL NEON data for Stage 2 fine-tune")
        ref_train_pairs = all_train
        ref_val_pairs = all_val

    # Load Stage 1 model as starting point
    processor_s2 = SegformerImageProcessor.from_pretrained(stage1_model)
    processor_s2.do_reduce_labels = False
    ref_class_weights = None if args.no_class_weights else compute_class_weights(ref_train_pairs)

    stage2_model = run_stage(
        stage_name="Stage 2: NEON Fine-tune",
        train_pairs=ref_train_pairs,
        val_pairs=ref_val_pairs,
        processor=processor_s2,
        output_dir=STAGE2_DIR,
        epochs=config.FINETUNE_EPOCHS,
        lr=config.FINETUNE_LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        grad_accum=config.GRAD_ACCUM_STEPS,
        class_weights=ref_class_weights,
    )

    # =========================================================================
    # Copy final model to segformer_tree_crown/
    # =========================================================================
    print("\n" + "=" * 60)
    print("COPYING FINAL MODEL")
    print("=" * 60)

    os.makedirs(FINAL_DIR, exist_ok=True)
    for fname in os.listdir(stage2_model):
        src = os.path.join(stage2_model, fname)
        dst = os.path.join(FINAL_DIR, fname)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"  Copied -> {FINAL_DIR}")

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE!")
    print("=" * 60)
    print(f"  Stage 1: {STAGE1_DIR}")
    print(f"  Stage 2: {STAGE2_DIR}")
    print(f"  Final:   {FINAL_DIR}")


if __name__ == "__main__":
    main()
