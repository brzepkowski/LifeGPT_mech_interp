"""
Attention pattern visualization for LifeGPT.

Selects a single moderate-entropy example (live-cell density ≈ 0.35),
runs a teacher-forced forward pass with flash attention disabled so that
full attention weight matrices are available, then saves three plot types:

  plots/attn_example_grid.png       — the chosen input/output GoL grids
  plots/attn_full_layer_{k:02d}.png — downsampled full attention matrix, all 8 heads
  plots/attn_field_layer_{k:02d}.png— spatial attention "field" from 4 output cells
  plots/attn_summary.png            — per-head entropy by layer (summary heatmap)
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
print(device)

# ── Constants ────────────────────────────────────────────────────────────────

NUM_WORDS      = 256
MAX_LENGTH     = len("@PredictNextState<> []$") + (32 * 32 * 2)   # 2071
GRID_SIZE      = 32
N_SEARCH       = 500    # scan this many rows to find a good example
TARGET_DENSITY = 0.35   # target live-cell fraction (moderate entropy)
DOWNSAMPLE     = 8      # factor for full-attention matrix thumbnails
MIN_BORDER_ALIVE = 3    # minimum live cells on the grid border to select an example

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

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

# ── Helper classes ────────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, n_pad: int, device: torch.device, pad_byte: int = 0):
        self.n_pad    = n_pad
        self.device   = device
        self.pad_byte = pad_byte

    def tokenize_str(self, sentence: str, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert len(base) == self.n_pad
        return torch.Tensor(base).long().to(self.device)

    def texts_to_sequences(self, texts: typing.List[str], encoding="utf8", do_padding=True):
        seqs = [self.tokenize_str(s, do_padding=do_padding).unsqueeze(0) for s in texts]
        return torch.cat(seqs, dim=0).to(self.device)


def neighbor_sum(grid: np.ndarray) -> np.ndarray:
    """8-connected neighbor count per cell (toroidal). Returns int32 array 0–8."""
    return sum(
        np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
    ).astype(np.int32)


def gol_step(grid: np.ndarray) -> np.ndarray:
    n = neighbor_sum(grid)
    return ((n == 3) | (grid.astype(bool) & (n == 2))).astype(np.int32)

# ── Load model (flash attention OFF so weights are computed) ──────────────────

def load_model(model_path, max_length, num_words):
    torch.cuda.empty_cache()
    model = TransformerWrapper(
        num_tokens=num_words,
        max_seq_len=max_length,
        attn_layers=Decoder(
            dim=256, depth=12, heads=8, attn_dim_head=64,
            rotary_pos_emb=True,
            attn_flash=False,   # must be False to capture attention weights
        ),
    )
    model = AutoregressiveWrapper(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    print("Model loaded (flash attention disabled for weight capture)")
    return model


model     = load_model(MODEL_PATH, MAX_LENGTH, NUM_WORDS)
tokenizer = Tokenizer(MAX_LENGTH, device)

# ── Select moderate-entropy example ──────────────────────────────────────────

print(f"\nSearching first {N_SEARCH} rows for density ≈ {TARGET_DENSITY} "
      f"with ≥ {MIN_BORDER_ALIVE} live border cells ...")
df = pd.read_csv(DATA_CSV, nrows=N_SEARCH, dtype=str)

# Pre-compute border cell indices for a GRID_SIZE×GRID_SIZE flattened state
_G = GRID_SIZE
_border_idx = np.array(sorted(set(
    list(range(_G))                                    # top row
    + list(range((_G - 1) * _G, _G * _G))             # bottom row
    + list(range(0, _G * _G, _G))                      # left column
    + list(range(_G - 1, _G * _G, _G))                # right column
)))


def _parse_state(s: str) -> np.ndarray:
    return np.array(list(s), dtype=np.int32)


densities    = np.empty(len(df), dtype=float)
border_alive = np.empty(len(df), dtype=int)
for i, (_, row) in enumerate(df.iterrows()):
    cells           = _parse_state(row["State 1"])
    densities[i]    = cells.mean()
    border_alive[i] = cells[_border_idx].sum()

border_mask = border_alive >= MIN_BORDER_ALIVE
if not border_mask.any():
    raise RuntimeError(
        f"No example with ≥ {MIN_BORDER_ALIVE} live border cells found "
        f"in the first {N_SEARCH} rows. Increase N_SEARCH or lower MIN_BORDER_ALIVE."
    )

best_idx      = int(np.argmin(np.abs(densities - TARGET_DENSITY) + (~border_mask) * 1e9))
row           = df.iloc[best_idx]
current_state = np.array(list(row["State 1"]), dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
next_state    = np.array(list(row["State 2"]), dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
density       = densities[best_idx]
print(f"Selected row {best_idx}: density = {density:.3f}  "
      f"({current_state.sum()} / {GRID_SIZE*GRID_SIZE} live cells, "
      f"{border_alive[best_idx]} live border cells)")

# ── Build full input sequence ─────────────────────────────────────────────────

state_str     = "".join(str(int(c)) for c in current_state.flatten())
state_str_out = "".join(str(int(c)) for c in next_state.flatten())
full_seq      = f"@PredictNextState<{state_str}> [{state_str_out}]$"
tokens        = tokenizer.texts_to_sequences([full_seq], do_padding=False)

input_start  = full_seq.index('<') + 1         # first cell of input grid in sequence
input_end    = input_start + GRID_SIZE * GRID_SIZE
output_start = full_seq.index('[') + 1         # first cell of output grid
output_end   = output_start + GRID_SIZE * GRID_SIZE
seq_len      = tokens.shape[1]

print(f"Sequence length : {seq_len}")
print(f"Input  positions: {input_start} – {input_end-1}")
print(f"Output positions: {output_start} – {output_end-1}")

# ── Run forward pass and capture attention weights ────────────────────────────

print("\nRunning forward pass ...")
with torch.no_grad():
    embedded = model.net.token_emb(tokens)                           # (1, seq_len, d_model)
    _, hiddens = model.net.attn_layers(embedded, return_hiddens=True)

# Move all attention matrices to CPU immediately and discard GPU tensors
attn_list = [
    inter.post_softmax_attn[0].detach().cpu().numpy()  # (n_heads, seq_len, seq_len)
    for inter in hiddens.attn_intermediates
]
del hiddens

n_layers = len(attn_list)
n_heads  = attn_list[0].shape[0]
print(f"Captured {n_layers} layers × {n_heads} heads, "
      f"each matrix: {attn_list[0].shape[1]}×{attn_list[0].shape[2]}")

# ── Plot: layer 0, full-resolution attention matrix, one file per head ────────
#
# The full seq_len×seq_len matrix is rendered at ~1 px/token (no explicit
# pooling).  Saving one file per head keeps each image at a manageable size and
# lets the viewer zoom in to individual token pairs.
#
# Region markers:
#   red  dashed  = start of input grid   (position input_start)
#   red  dotted  = end   of input grid   (position input_end)
#   cyan dashed  = start of output grid  (position output_start)
#   cyan dotted  = end   of output grid  (position output_end)

print("\nSaving full-resolution layer-0 attention (one file per head) ...")
attn0 = attn_list[0]   # (n_heads, seq_len, seq_len)

_dpi  = 100
_side = seq_len / _dpi   # 1 px per token at this DPI

for h in range(n_heads):
    fig, ax = plt.subplots(figsize=(_side, _side))
    vmax = np.percentile(attn0[h], 99.5)
    ax.imshow(attn0[h], aspect="equal", cmap="viridis",
              vmin=0, vmax=vmax, interpolation="nearest")

    for pos, color, ls in [(input_start,  "red",  "--"),
                            (input_end,    "red",  ":"),
                            (output_start, "cyan", "--"),
                            (output_end,   "cyan", ":")]:
        ax.axhline(pos, color=color, lw=0.6, linestyle=ls)
        ax.axvline(pos, color=color, lw=0.6, linestyle=ls)

    ax.set_title(
        f"Layer 0  Head {h}  —  {seq_len}×{seq_len}, 1 px/token | red=input  cyan=output",
        fontsize=8,
    )
    ax.set_xlabel("key token index", fontsize=7)
    ax.set_ylabel("query token index", fontsize=7)

    plt.tight_layout(pad=0.3)
    out_path = os.path.join(PLOTS_DIR, f"attn_layer00_head{h:02d}_full.png")
    plt.savefig(out_path, dpi=_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Head {h}: saved {os.path.basename(out_path)}")

print("\nDone.")

# ── Plot 0: example grid ──────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(current_state, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
axes[0].set_title(f"Input state  ({current_state.sum()} live cells, density={density:.2f})")
axes[0].axis("off")

axes[1].imshow(next_state, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title("Output state (GoL step)")
axes[1].axis("off")

ns = neighbor_sum(current_state)
im = axes[2].imshow(ns, cmap="YlOrRd", vmin=0, vmax=8, interpolation="nearest")
axes[2].set_title("Neighbor sum (ground truth target)")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], ticks=range(9))

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "attn_example_grid.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {os.path.basename(out_path)}")
