"""
2_2_replot_probe_predictions.py

Re-generates best_probe_prediction_vs_actual.png from saved probe_results.pt
without rerunning probe training.  Adds cell-level grid lines and row/column
indices to every subplot.

Example A (high-entropy grid): fully derived from probe_results.pt + CSV.
Example B (sparse grid):       hardcoded grid; requires one forward pass
                                through the model to get activations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import typing

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE  = 32
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

DATA_CSV = os.path.join(
    SCRIPT_DIR,
    "../LifeGPT/conway_states_0_1_10000by32by32by10_toroidal_20240711_133408.csv",
)
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "../model_parameters/"
    "07_22_2024_Conway_2_State_Jump_Rot_Pos_On_Masking_On_Broad_Entropy_Homog_"
    "2024-07-23 10-37-31/LifeGPT_epoch_50.pt",
)
PROBE_PATH = os.path.join(SCRIPT_DIR, "activations", "probe_results.pt")

# ── Load saved probe results ──────────────────────────────────────────────────

print("Loading probe_results.pt ...")
saved       = torch.load(PROBE_PATH)
results     = saved["results"]
all_acts_np = saved["all_acts"]          # (N_SAMPLES, n_ckpts, 1024, d_model)
all_nbrs_np = saved["all_neighbors"]     # (N_SAMPLES, 1024)
N_TRAIN     = saved["N_TRAIN"]

best        = max(results, key=lambda r: r["test_acc"])
best_r      = results[best["checkpoint"]]
print(f"Best probe: checkpoint {best['checkpoint']}  test_acc={best['test_acc']:.3f}")

# ── Helper: apply a saved probe ───────────────────────────────────────────────

def probe_predict(r: dict, X: np.ndarray) -> np.ndarray:
    Xt = (torch.from_numpy(X.astype(np.float32)) - r["mean"]) / r["std"]
    return (Xt @ r["W"] + r["b"]).argmax(dim=-1).numpy()

# ── Example A: high-entropy grid (from saved activations) ────────────────────

print("Reconstructing Example A from saved data ...")
TOTAL_ROWS = 10000
N_SAMPLES  = all_acts_np.shape[0]
sample_idx = np.linspace(0, TOTAL_ROWS - 1, N_SAMPLES, dtype=int)
rows_to_skip = sorted(set(range(1, TOTAL_ROWS + 1)) - set(sample_idx + 1))
df = pd.read_csv(DATA_CSV, skiprows=rows_to_skip, dtype=str)

test_grid   = np.array(list(df.iloc[N_TRAIN]["State 1"]),
                        dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
test_acts   = all_acts_np[N_TRAIN, best["checkpoint"], :, :]
test_pred   = probe_predict(best_r, test_acts).reshape(GRID_SIZE, GRID_SIZE)
test_actual = all_nbrs_np[N_TRAIN].reshape(GRID_SIZE, GRID_SIZE)
test_err    = (test_pred != test_actual).astype(int)

# ── Example B: sparse grid (one forward pass, no training) ───────────────────

print("Running single forward pass for Example B ...")

class Tokenizer:
    def __init__(self, n_pad, device, pad_byte=0):
        self.n_pad = n_pad; self.device = device; self.pad_byte = pad_byte

    def tokenize_str(self, sentence, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            base.extend([self.pad_byte] * (self.n_pad - len(base)))
        return torch.Tensor(base).long().to(self.device)

    def texts_to_sequences(self, texts, encoding="utf8", do_padding=True):
        return torch.cat(
            [self.tokenize_str(s, do_padding=do_padding).unsqueeze(0) for s in texts]
        ).to(self.device)


def neighbor_sum(grid):
    return sum(
        np.roll(np.roll(grid, dr, 0), dc, 1)
        for dr in [-1, 0, 1] for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
    ).astype(np.int32)


MAX_LENGTH = len("@PredictNextState<> []$") + (GRID_SIZE * GRID_SIZE * 2)
tokenizer  = Tokenizer(MAX_LENGTH, device)

model = TransformerWrapper(
    num_tokens=256, max_seq_len=MAX_LENGTH,
    attn_layers=Decoder(dim=256, depth=12, heads=8, attn_dim_head=64,
                        rotary_pos_emb=True, attn_flash=True),
)
model = AutoregressiveWrapper(model)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device); model.eval()

_norm_cls = (torch.nn.LayerNorm,)
try:
    from x_transformers.x_transformers import RMSNorm, ScaleNorm
    _norm_cls = (*_norm_cls, RMSNorm, ScaleNorm)
except Exception:
    pass

simple_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
for r, c in [(0,1),(1,2),(2,0),(2,1),(2,2)]:
    simple_grid[r+2, c+2] = 1
for r, c in [(0,0),(0,1),(0,2)]:
    simple_grid[r+15, c+14] = 1
for r, c in [(0,0),(0,1),(1,0),(1,1)]:
    simple_grid[r+25, c+25] = 1

simple_next = ((neighbor_sum(simple_grid) == 3) |
               (simple_grid.astype(bool) & (neighbor_sum(simple_grid) == 2))).astype(np.int32)

state_str     = "".join(str(int(c)) for c in simple_grid.flatten())
state_str_out = "".join(str(int(c)) for c in simple_next.flatten())
full_seq      = f"@PredictNextState<{state_str}> [{state_str_out}]$"
tokens        = tokenizer.texts_to_sequences([full_seq], do_padding=False)
output_start  = full_seq.index('[') + 1

_acts_b: dict = {}
_hooks = []

def _embed_hook(module, inputs, output):
    _acts_b["00_embed"] = output.detach().float().cpu()
_hooks.append(model.net.token_emb.register_forward_hook(_embed_hook))

def _make_pre_hook(name):
    def hook(module, inputs):
        _acts_b[name] = inputs[0].detach().float().cpu()
    return hook
for name, module in model.net.attn_layers.named_modules():
    if isinstance(module, _norm_cls):
        _hooks.append(module.register_forward_pre_hook(_make_pre_hook(name)))

with torch.no_grad():
    model.net(tokens)
for h in _hooks:
    h.remove()
del model

keys_b      = list(_acts_b.keys())
simple_acts = torch.stack([_acts_b[k].squeeze(0) for k in keys_b], dim=0)
simple_acts = simple_acts[:, output_start:output_start + GRID_SIZE*GRID_SIZE, :]

simple_actual = neighbor_sum(simple_grid)
simple_pred   = probe_predict(
                    best_r, simple_acts[best["checkpoint"]].numpy()
                ).reshape(GRID_SIZE, GRID_SIZE)
simple_err    = (simple_pred != simple_actual).astype(int)

# ── Plot helper ───────────────────────────────────────────────────────────────

def apply_grid_and_ticks(ax):
    """Add cell-level grid lines and row/column indices to an imshow axis."""
    ticks      = range(0, GRID_SIZE, 4)   # label every 4th cell
    tick_minor = np.arange(-0.5, GRID_SIZE, 1)

    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels([str(t) for t in ticks], fontsize=5)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=5)
    ax.tick_params(axis="both", length=2, pad=1)

    ax.set_xticks(tick_minor, minor=True)
    ax.set_yticks(tick_minor, minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3, alpha=0.6)
    ax.tick_params(which="minor", length=0)

    ax.set_xlabel("column", fontsize=6, labelpad=2)
    ax.set_ylabel("row",    fontsize=6, labelpad=2)

# ── Draw ──────────────────────────────────────────────────────────────────────

examples = [
    ("High-entropy grid",
     test_grid, test_actual, test_pred, test_err),
    ("Sparse pattern (glider + blinker + block)",
     simple_grid, simple_actual, simple_pred, simple_err),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 11))
cmap, vmin, vmax = "YlOrRd", 0, 8

for row, (label, grid, actual, pred, err) in enumerate(examples):
    im0 = axes[row, 0].imshow(grid, cmap="binary", vmin=0, vmax=1,
                               interpolation="nearest")
    axes[row, 0].set_title(f"{label}\n({grid.sum()} live cells)", fontsize=10)
    plt.colorbar(im0, ax=axes[row, 0], ticks=[0, 1])
    apply_grid_and_ticks(axes[row, 0])

    im1 = axes[row, 1].imshow(actual, cmap=cmap, vmin=vmin, vmax=vmax,
                               interpolation="nearest")
    axes[row, 1].set_title("Actual neighbor sum", fontsize=10)
    plt.colorbar(im1, ax=axes[row, 1], ticks=range(9))
    apply_grid_and_ticks(axes[row, 1])

    im2 = axes[row, 2].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax,
                               interpolation="nearest")
    axes[row, 2].set_title(
        f"Predicted (ckpt {best['checkpoint']})\nacc = {best['test_acc']:.3f}",
        fontsize=10)
    plt.colorbar(im2, ax=axes[row, 2], ticks=range(9))
    apply_grid_and_ticks(axes[row, 2])

    im3 = axes[row, 3].imshow(err, cmap="Reds", vmin=0, vmax=1,
                               interpolation="nearest")
    axes[row, 3].set_title(
        f"Errors  ({err.sum()} / {GRID_SIZE*GRID_SIZE} cells wrong)", fontsize=10)
    plt.colorbar(im3, ax=axes[row, 3])
    apply_grid_and_ticks(axes[row, 3])

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "best_probe_prediction_vs_actual.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_path}")
