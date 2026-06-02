# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LifeGPT is a decoder-only GPT (via `x_transformers`) trained to simulate Conway's Game of Life on a toroidal grid. It operates as a next-state predictor: given a flattened 2D grid state as a token sequence, it autoregressively generates the next state. The research explores topology-agnostic simulation and an "autoregressive autoregressor" (ARAR) loop where model outputs are fed back as inputs for multi-step rollouts.

## Environment setup

The original `LifeGPT_env.yml` was exported from Windows and will not work on Linux. Use the Linux-compatible file instead:

```bash
conda env create -f LifeGPT_env_linux.yml
conda activate LifeGPT_env
```

There is no build system, test suite, or linter configured.

## Running training

```bash
# With forgetful causal masking (FCM), broad-entropy data — best performing model
python LifeGPT/LifeGPT_toroidal_rot_pos_on_15_percent_forgetful_mask_broad_ent.py

# Without FCM, high-entropy data
python LifeGPT/LifeGPT_toroidal_rot_pos_on_no_forgetful_mask_high_ent.py
```

Both scripts train, periodically benchmark accuracy, and save `.pt` checkpoints under `model_parameters/<model_name>_<timestamp>/`.

## Key notebooks

| Notebook | Purpose |
|---|---|
| `training_set_gen.ipynb` | Generate train/val/test CSV datasets |
| `Accuracy_Benchmarking.ipynb` | Load a checkpoint and measure per-cell accuracy across epochs and temperatures |
| `ARAR_9_iterations.ipynb` / `ARAR_249_iterations.ipynb` | Recursive multi-step rollout (ARAR loop) |
| `Training_Data_Ordering_Investigation.ipynb` | Entropy-based analysis of training data ordering effects |
| `Multi_Grid/` | Variants trained simultaneously on multiple grid sizes |

## Architecture

### Sequence format

All data is serialized as ASCII strings and tokenized as raw UTF-8 bytes (`num_words=256`). A training sample for a 32×32 grid looks like:

```
@PredictNextState<{1024 chars of 0/1}> [{1024 chars of 0/1}]$
```

- `<...>` contains the current game state (flattened row-major)
- `[...]` contains the ground-truth next state
- `max_length = 2071` for 32×32 grids; adjust for other sizes

The prompt fed to the model at inference time is everything up to and including `>` (the task prefix). The model then autoregressively generates the `[...]` portion.

### Model

Built with `x_transformers.TransformerWrapper` + `AutoregressiveWrapper`:

```python
TransformerWrapper(
    num_tokens=256,          # byte-level vocabulary
    max_seq_len=max_length,
    attn_layers=Decoder(
        dim=256, depth=12, heads=8, attn_dim_head=64,
        rotary_pos_emb=True,   # RoPE positional encoding
        attn_flash=True        # Flash attention
    )
)
```

FCM (forgetful causal masking) is applied at the `AutoregressiveWrapper` level via `mask_prob=0.15`.

### `conway_lib/` package

- `game.py` — `ConwayGame` class: grid simulation (toroidal and zero-BC), dataset generation (`generate_sets`), tunable IC generation via Monte Carlo heterogeneity adjustment, and named pattern initialization (`glider_gun`, `cloverleaf`, etc.). **Note:** `game.py` contains runnable code at module level (lines 509–513) that executes on import — this will launch a matplotlib animation window whenever the module is imported.
- `tokenizer.py` — `Tokenizer`: UTF-8 byte tokenization with padding; `texts_to_sequences` and `sequences_to_texts`.
- `model.py` — `ConwayModel`: wrapper around the x_transformers model with checkpoint directory creation.
- `optimizer.py` — `Optimizer`: Bayesian hyperparameter optimization using `scikit-optimize`.
- `__init__.py` — only exports `ConwayGame`; `Tokenizer`, `ConwayModel`, and `Optimizer` are commented out and must be imported directly from their modules.

### Data files

CSV files in `LifeGPT/` follow the naming convention:
```
conway_states_{s_order}_{e_order}_{A}by{N}by{N}by{I}_toroidal_{timestamp}.csv
```
where `A` = number of samples, `N` = grid size, `I` = number of timesteps, `s/e_order` = entropy range (0→1 = broad, 0.5→0.5 = high-entropy).

## Inference pattern

```python
# Load model
model = AutoregressiveWrapper(TransformerWrapper(...))
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Tokenize prompt (everything up to and including '>')
inp = extract_task(input_seq, end_task_token='>')
inp = torch.Tensor(tokenizer.texts_to_sequences(inp, do_padding=False)).to(device)
inp = inp.transpose(0, 1).long()

# Generate
with torch.no_grad():
    sample = model.generate(prompts=inp, seq_len=generate_length, temperature=temp, cache_kv=True)

output_str = tokenizer.sequences_to_texts(sample[:1])
pred = extract_start_and_end(output_str[0], start_token='[', end_token=']')
```
