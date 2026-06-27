# ARM-ARC-AGI

Research codebase for the Accelerated Recursive Reasoning Model (ARM), a hierarchical transformer architecture evaluated on ARC-AGI.

## What this is

ARC-AGI tasks present a small set of input-output grid pairs and ask a model to derive the output for a held-out input. Each task encodes a different visual transformation rule. The model must identify the rule from a handful of examples and apply it.

This repo trains and evaluates two model families:

**TRM** (Tiny Recursive Reasoning Model): a two-level recurrent transformer. An L-level network processes the full 30x30 grid at original resolution. An H-level network operates on a spatially compressed (15x15) representation. The two levels alternate for several cycles before producing a prediction. Task identity is injected via a learned per-puzzle embedding.

**ARM** (Accelerated Recursive Reasoning Model): extends TRM by replacing the H-level spatial tokens with a fixed set of 32 learned latent tokens. The L- and H-levels interact through bidirectional cross-attention: L-tokens attend to the latents to receive abstract guidance (broadcast), then H-latents attend back to L-tokens to extract updated information (perceive). The halting mechanism from Adaptive Computation Time (Graves, 2016) decides how many cycles to run based on the mean-pooled latent state.

The key question is whether Perceiver-style latent abstraction improves generalisation over the baseline TRM at matched parameter budgets.

## Repository layout

```
pretrain.py                  main training entry point
evaluate_checkpoint.py       standalone evaluator (single checkpoint or watch mode)
puzzle_dataset.py            dataset loader shared by all architectures

config/
  cfg_pretrain.yaml          base training hyperparameters
  arch/                      per-architecture Hydra config files

utils/
  models/
    recursive_reasoning/     model implementations (trm.py, trm_abstraction.py, ...)
    layers.py                shared building blocks (attention, RoPE, SwiGLU, ...)
    sparse_embedding.py      per-puzzle embedding with SignSGD
  dataset/
    build_arc_dataset.py     dataset builder (ARC-AGI-1, ConceptARC, RE-ARC)
    build_arc_agi_2_dataset.py
    raw-data/                git submodules for ARC-AGI-1/2, ConceptARC, RE-ARC

experiments/                 output directory (checkpoints, eval JSON, wandb run id)
scripts/
  install.sh                 environment bootstrap (also a valid sbatch script)
  dataset/                   dataset build shell scripts
```

## Setup

Python 3.11 or newer is required. The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# clone with submodules (ARC-AGI-1/2, ConceptARC, RE-ARC)
git clone --recurse-submodules <repo-url>
cd arm-arc-agi

# install dependencies
bash scripts/install.sh
```

On a SLURM cluster the same script can be submitted as an sbatch job. It will detect `SLURM_SUBMIT_DIR` automatically.

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:

| Variable | Purpose |
|---|---|
| `WANDB_ENTITY` | W&B username or team |
| `WANDB_PROJECT` | W&B project name |
| `WANDB_MODE` | `online` or `disabled` |
| `DATALOADER_NUM_WORKERS` | must be `1` (hard requirement in puzzle_dataset.py) |

## Building the dataset

ARC-AGI-1 with 1000 augmentations per puzzle:

```bash
uv run python utils/dataset/build_arc_dataset.py \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --output-dir data/arc-aug-1000 \
  --augmentations 1000
```

Or use the provided shell script:

```bash
bash scripts/dataset/dataset-arc-agi.sh
```

The builder reads from `utils/dataset/raw-data/` (populated by the git submodules) and writes a processed dataset to `data/`.

## Training

Training uses Hydra for configuration. The architecture is selected by overriding the `arch` group.

Train ARM (the main model):

```bash
uv run python pretrain.py arch=trm_abstraction
```

Train the TRM baseline:

```bash
uv run python pretrain.py arch=trm
```

Common overrides:

```bash
# change learning rate and batch size
uv run python pretrain.py arch=trm_abstraction lr=3e-4 global_batch_size=512

# resume from a checkpoint
uv run python pretrain.py arch=trm_abstraction load_checkpoint=experiments/my-run/checkpoints/step_50000

# disable W&B
uv run python pretrain.py arch=trm_abstraction WANDB_MODE=disabled
```

Checkpoints and evaluation results are written under `experiments/<run-name>/`.

Multi-GPU training uses `torchrun`:

```bash
torchrun --nproc_per_node=4 pretrain.py arch=trm_abstraction
```

## Evaluation

`evaluate_checkpoint.py` is architecture-agnostic. It reads the saved `all_config.yaml` from the checkpoint directory to reconstruct the model.

```bash
# evaluate a single checkpoint
uv run python evaluate_checkpoint.py \
  --checkpoint experiments/my-run/checkpoints/step_50000

# watch a checkpoint directory and evaluate as new checkpoints appear
uv run python evaluate_checkpoint.py \
  --watch experiments/my-run/checkpoints \
  --poll-interval 60

# evaluate on a different dataset than the one used for training
uv run python evaluate_checkpoint.py \
  --checkpoint experiments/my-run/checkpoints/step_50000 \
  --data-path data/arc2concept-aug-1000

# multi-worker parallel evaluation
uv run python evaluate_checkpoint.py \
  --watch-root experiments \
  --worker-id 0 --num-workers 4
```

The primary metrics are `ARC/pass@1` (sampled, 32 attempts per puzzle) and `all/exact_accuracy` (greedy single prediction).

## Architecture details

### Shared components

Both TRM and ARM share the following:

- L-level: 2 transformer layers, hidden size 512, 8 heads, 30x30 grid, RoPE positional encodings
- Per-puzzle embedding: learned sparse embedding injected into each cycle via addition
- Halting: ACT sigmoid halt signal, up to 16 steps, with an exploration probability of 0.1 during training
- Loss: stablemax cross-entropy on the output token distribution

### TRM

The H-level compresses the 30x30 L-output to 15x15 via stride-2 pooling. H operates with 3 transformer layers, hidden size 512, 8 heads. Three H-cycles and six L-cycles run per forward pass. Total parameters: approximately 6.8M.

### ARM

The H-level replaces spatial tokens with 32 learned latent tokens (hidden size 256, 3 layers, 4 heads). Each H-cycle consists of four steps: H-to-L broadcast cross-attention, L self-attention (4 cycles), L-to-H perceive cross-attention, and H self-attention. The latent tokens carry no positional encoding. Two H-cycles and four L-cycles run per forward pass. Total parameters: approximately 10.8M.

## Key results

At 32M training samples, ARM reaches approximately 4.5 percentage points above the TRM baseline on `ARC/pass@1`. Extended runs to 58M samples show all three variants (TRM-Opt, broadcast-only, gated) converging to ARM's 25.0% ceiling, confirming that the broadcast and gate mechanisms are gradient-equivalent at scale and that the gain comes from the latent abstraction structure rather than the conditioning mechanism.

## Dependencies

Notable packages: PyTorch 2.7, Triton 3.3 (Linux only), Hydra-Core 1.3, W&B 0.26, Pydantic 2.11, adam-atan2-pytorch, einops, numba.

Full dependency list: `pyproject.toml` / `uv.lock`.
