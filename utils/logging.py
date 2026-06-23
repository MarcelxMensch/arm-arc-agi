"""Training logging utilities for wandb.

Provides grid visualization, embedding analysis, gradient norm monitoring,
and Q-value precision/recall analysis.

Logging frequency:
  every step:        loss, accuracy, lr, pics_per_sec, grad_norm, q_precision, q_recall, q_f1
  (Call ``mark_step_compute_done()`` after the training forward/backward/optim work so
  ``pics_per_sec`` excludes extra instrumentation such as full-ACT probes.)
  every 1000 steps:  grid images (3 examples), puzzle embedding PCA
  every eval:        confusion matrix, eval grid images, PCA snapshot,
                     Q-value confidence histogram, per-task Q-value table
"""
import os
import json
import time
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import wandb
from PIL import Image

from utils.dataset.build_arc_dataset import inverse_aug


# ═══════════════════════════════════════════════════════════════════════════════
# ARC Color Palette & Grid Visualization
# ═══════════════════════════════════════════════════════════════════════════════

# Standard ARC 10-color palette (indices 0-9)
ARC_PALETTE = np.array([
    [0,   0,   0],    # 0: Black
    [0,   116, 217],  # 1: Blue
    [255, 65,  54],   # 2: Red
    [46,  204, 64],   # 3: Green
    [255, 220, 0],    # 4: Yellow
    [170, 170, 170],  # 5: Gray
    [240, 18,  190],  # 6: Magenta
    [255, 133, 27],   # 7: Orange
    [127, 219, 255],  # 8: Cyan
    [135, 12,  37],   # 9: Maroon
], dtype=np.uint8)


def colorize_grid(grid: np.ndarray, cell_size: int = 8) -> np.ndarray:
    """Convert grid (values 0-9) to RGB numpy array, scaled by cell_size."""
    h, w = grid.shape
    img = ARC_PALETTE[grid.clip(0, 9)]  # [h, w, 3]
    return np.repeat(np.repeat(img, cell_size, axis=0), cell_size, axis=1)


def seq_to_grid_30x30(seq: np.ndarray, cell_size: int = 8) -> np.ndarray:
    """Convert flat 900-token sequence to a 30x30 colored RGB array (always full grid)."""
    grid = seq[:900].reshape(30, 30)
    return colorize_grid(grid, cell_size)


def make_composite_image(input_seq, pred_seq, target_seq, cell_size=8):
    """Create vertical composite: input (top) / target (middle) / prediction (bottom).

    Each grid is rendered as a full 30x30, giving a 30x90 cell layout.
    A 2-pixel white dividing line separates the three panels.
    """
    inp_img = seq_to_grid_30x30(input_seq, cell_size)
    tgt_img = seq_to_grid_30x30(target_seq, cell_size)
    pred_img = seq_to_grid_30x30(pred_seq, cell_size)

    w = inp_img.shape[1]  # all are 30*cell_size wide
    divider = np.full((2, w, 3), 255, dtype=np.uint8)  # white line

    composite = np.concatenate([
        inp_img, divider,
        tgt_img, divider,
        pred_img,
    ], axis=0)
    return Image.fromarray(composite)


# ═══════════════════════════════════════════════════════════════════════════════
# Puzzle ID Resolution
# ═══════════════════════════════════════════════════════════════════════════════

def load_identifier_maps(data_paths: List[str]) -> List[list]:
    """Load identifiers.json from all data paths."""
    maps = []
    for path in data_paths:
        id_path = os.path.join(path, "identifiers.json")
        if os.path.isfile(id_path):
            with open(id_path) as f:
                maps.append(json.load(f))
    return maps


def get_arc_puzzle_id(identifier_maps: List[list], internal_id: int) -> str:
    """Resolve internal puzzle identifier to original ARC puzzle ID.

    Strips augmentation info (dihedral transform, color permutation) to return
    the base puzzle name usable at https://arcprize.org/play?task=<id>.
    """
    for id_map in identifier_maps:
        if internal_id < len(id_map):
            name = id_map[internal_id]
            if name == "<blank>":
                return "<blank>"
            orig_name, _ = inverse_aug(name)
            return orig_name
    return f"unknown_{internal_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# Model Access Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _unwrap_to_inner(model):
    """Navigate ACTLossHead -> ACT wrapper -> Inner, handling torch.compile."""
    m = model
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    if hasattr(m, 'model'):  # ACTLossHead -> ACT wrapper
        m = m.model
    if hasattr(m, 'inner'):  # ACT wrapper -> inner
        return m.inner
    return None



# ═══════════════════════════════════════════════════════════════════════════════
# TrainLogger - Orchestrates all logging
# ═══════════════════════════════════════════════════════════════════════════════

class TrainLogger:
    """Comprehensive training logger for wandb.

    Usage in pretrain.py:
        logger = TrainLogger(model, config)
        for batch in train_loader:
            logger.step_start()
            # ... train_batch() with logger.compute_grad_metrics() inside ...
            logger.log_train_step(step, metrics, batch_cuda, carry, extra_outputs, gbs)
    """

    def __init__(self, model, config, log_interval_medium=100, log_interval_heavy=1000):
        self.model = model
        self.config = config
        self.identifier_maps = load_identifier_maps(config.data_paths)
        self.log_interval_medium = log_interval_medium
        self.log_interval_heavy = log_interval_heavy
        self._step_t0 = None
        self._step_compute_t1: Optional[float] = None

    # ─── Timing ───────────────────────────────────────────────────────────

    def step_start(self):
        """Call before each training step."""
        self._step_t0 = time.time()
        self._step_compute_t1 = None

    def mark_step_compute_done(self) -> None:
        """Record wall time after forward/backward/optimizer; excludes later logging-only work."""
        self._step_compute_t1 = time.time()

    def pics_per_sec(self, global_batch_size: int) -> float:
        if self._step_t0 is None:
            return 0.0
        t1 = self._step_compute_t1 if self._step_compute_t1 is not None else time.time()
        return global_batch_size / max(t1 - self._step_t0, 1e-6)

    # ─── Gradient Health ──────────────────────────────────────────────────

    def compute_grad_metrics(self) -> Dict[str, float]:
        """Compute global gradient norm on GPU (no CPU sync)."""
        total_norm_sq = torch.zeros(1, device="cuda")
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().float().pow(2).sum()
        return {"train/grad_norm": total_norm_sq.sqrt().item()}

    # ─── Main Logging Entry Points ────────────────────────────────────────

    def log_train_step(self, step, metrics, batch_cuda, carry, extra_outputs, global_batch_size):
        """Main per-step logging. Calls sub-loggers at appropriate intervals."""
        metrics["train/pics_per_sec"] = self.pics_per_sec(global_batch_size)

        if step % self.log_interval_medium == 0:
            self._log_medium(step, carry, extra_outputs, batch_cuda)

        if step % self.log_interval_heavy == 0:
            self._log_heavy(step, carry, extra_outputs, batch_cuda)

    def _log_medium(self, step, carry, extra_outputs, batch_cuda):
        """Log at medium frequency (every 100 steps). Currently a no-op placeholder."""
        pass

    def _log_heavy(self, step, carry, extra_outputs, batch_cuda):
        """Log at heavy frequency (every 1000 steps)."""
        inner = _unwrap_to_inner(self.model)

        # Grid images: input / target / prediction
        if extra_outputs and "preds" in extra_outputs and batch_cuda is not None:
            self._log_grid_images(step, batch_cuda, extra_outputs["preds"], prefix="train")

    def log_eval(self, step, all_batches, all_preds, eval_q_data=None):
        """Log evaluation-specific metrics (confusion matrix, grid images, PCA, Q-value analysis)."""
        if not all_batches or not all_preds:
            return

        # Eval grid images from last batch
        last_batch = all_batches[-1]
        last_preds = all_preds[-1]
        if "preds" in last_preds:
            self._log_grid_images(step, last_batch, last_preds["preds"], prefix="eval")

        # Confusion matrix (aggregate all batches)
        all_labels = []
        all_predictions = []
        for batch, preds in zip(all_batches, all_preds):
            if "preds" in preds:
                all_labels.append(batch["labels"].cpu().numpy().flatten())
                all_predictions.append(preds["preds"].cpu().numpy().flatten())

        if all_labels:
            labels = np.concatenate(all_labels)
            predictions = np.concatenate(all_predictions)
            mask = labels != -100
            labels = labels[mask]
            predictions = predictions[mask]

            color_names = ["PAD", "EOS", "black", "blue", "red", "green",
                           "yellow", "gray", "magenta", "orange", "cyan", "maroon"]
            wandb.log({
                "eval/confusion": wandb.plot.confusion_matrix(
                    y_true=labels.tolist(),
                    preds=predictions.tolist(),
                    class_names=color_names[:12]
                )
            }, step=step)

        # Q-value precision/recall analysis
        if eval_q_data:
            self._log_q_value_analysis(step, eval_q_data)

    # ─── Q-Value ROC Analysis ────────────────────────────────────────────

    def _log_q_value_analysis(self, step, eval_q_data):
        """Log ROC curve and AUC for Q-halt decisions."""
        try:
            from sklearn.metrics import roc_curve, auc
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Aggregate all eval batches
        all_q = []
        all_correct = []
        for qd in eval_q_data:
            q_vals = qd["q_halt_logits"].float().numpy()
            labels = qd["labels"].numpy()
            preds = qd["preds"].numpy()

            mask = (labels != -100)
            is_correct = mask & (preds == labels)
            seq_correct = (is_correct.sum(-1) == mask.sum(-1)) & (mask.sum(-1) > 0)

            all_q.append(q_vals)
            all_correct.append(seq_correct)

        q_values = np.concatenate(all_q)
        correct = np.concatenate(all_correct).astype(int)

        if len(np.unique(correct)) < 2:
            return

        # ROC curve: Q-value as score, "correct" as ground truth
        # Higher Q → model more confident it solved the puzzle
        fpr, tpr, _ = roc_curve(correct, q_values)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Q-halt ROC (step {step})")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)

        wandb.log({
            "eval/q_roc": wandb.Image(Image.fromarray(buf)),
            "eval/q_auc": roc_auc,
        }, step=step)

    # ─── Grid Images ──────────────────────────────────────────────────────

    def _log_grid_images(self, step, batch_cuda, preds, prefix="train", max_samples=3):
        """Log input/target/prediction composite images."""
        inputs = batch_cuda["inputs"].cpu().numpy()
        labels = batch_cuda["labels"].cpu().numpy()
        predictions = preds.cpu().numpy()
        identifiers = batch_cuda["puzzle_identifiers"].cpu().numpy()

        n = min(max_samples, inputs.shape[0])
        images = []
        for i in range(n):
            pid = get_arc_puzzle_id(self.identifier_maps, int(identifiers[i]))
            if pid == "<blank>":
                continue
            try:
                img = make_composite_image(inputs[i], predictions[i], labels[i])
                images.append(wandb.Image(img, caption=f"{pid}  [top=input | mid=target | bot=pred]"))
            except Exception:
                continue

        if images:
            wandb.log({f"{prefix}/examples": images}, step=step)

    # ─── Puzzle Embedding PCA + K-Means Clustering ─────────────────────────

    def _log_puzzle_embedding_pca(self, step, puzzle_emb_weights):
        """Log PCA scatter colored by k-means clusters + silhouette plot."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        weights = puzzle_emb_weights.cpu().float().numpy()
        if weights.shape[0] <= 1:
            return

        norms = np.linalg.norm(weights[1:], axis=1)
        active_mask = norms > 1e-6
        if active_mask.sum() < 3:
            return

        active_weights = weights[1:][active_mask]
        active_indices = np.where(active_mask)[0] + 1

        # Deduplicate augmented variants → one embedding per base ARC puzzle ID
        seen = {}
        for i, idx in enumerate(active_indices):
            pid = get_arc_puzzle_id(self.identifier_maps, int(idx))
            if pid not in seen:
                seen[pid] = active_weights[i]

        if len(seen) < 4:
            return

        X = np.stack(list(seen.values()))

        # PCA to 2D for visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)

        # K-means clustering in full embedding space
        n_clusters = min(8, len(X) // 2)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(X)

        # PCA scatter colored by cluster
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
        ax.set_title(f"Puzzle Embedding PCA + K-Means (step {step}, k={n_clusters})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)

        wandb.log({"train/puzzle_pca": wandb.Image(Image.fromarray(buf))}, step=step)

        # Silhouette plot via wandb.sklearn
        try:
            wandb.sklearn.plot_silhouette(kmeans, X, labels)
        except Exception:
            pass
