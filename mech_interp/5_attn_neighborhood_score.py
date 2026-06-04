"""
6_attn_neighborhood_score.py  —  Per-head Moore-neighborhood attention score.

For every output cell (r, c) and each attention head in layer 0, computes the
fraction of input-region attention that falls on the cell's 8 toroidal Moore
neighbours (the ground-truth relevant cells for the GoL rule).

  score = sum(attn to the 8 Moore neighbours) / sum(attn to entire input block)

  1.0  → head perfectly routes all input-attention to the correct neighbours
  0.0  → head attends to completely wrong positions

If the model has separate "tools" for bulk vs. border cells:
  • bulk-specialist heads  → high score in the bulk, low at borders
  • border-specialist heads → low score in the bulk, high at borders/corners

Output files (mech_interp/plots/):
  attn_nbr_layer00_<tag>_heads.png  — 2×4 grid, one 32×32 heatmap per head
  attn_nbr_layer00_<tag>_mean.png   — mean score across all 8 heads + input grid
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_WORDS        = 256
MAX_LENGTH       = len("@PredictNextState<> []$") + (32 * 32 * 2)
GRID_SIZE        = 32
N_SEARCH         = 10_000
TARGET_DENSITY   = 0.35
MIN_BORDER_ALIVE = 3
LAYERS           = list(range(12))   # which transformer layers to analyse

# "dataset"   — average over N_EXAMPLES rows from the CSV
# "all_alive" — every cell is 1 (maximally symmetric, content-free baseline)
EXAMPLE_MODE = "dataset"
N_EXAMPLES   = 10   # number of dataset examples to average over (dataset mode only)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots", "attn")
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

# ── Model + tokenizer ─────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, n_pad, device, pad_byte=0):
        self.n_pad    = n_pad
        self.device   = device
        self.pad_byte = pad_byte

    def tokenize_str(self, sentence, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert len(base) == self.n_pad
        return torch.Tensor(base).long().to(self.device)

    def texts_to_sequences(self, texts, encoding="utf8", do_padding=True):
        seqs = [self.tokenize_str(s, do_padding=do_padding).unsqueeze(0) for s in texts]
        return torch.cat(seqs, dim=0).to(self.device)


def load_model(model_path, max_length, num_words):
    torch.cuda.empty_cache()
    model = TransformerWrapper(
        num_tokens=num_words,
        max_seq_len=max_length,
        attn_layers=Decoder(
            dim=256, depth=12, heads=8, attn_dim_head=64,
            rotary_pos_emb=True,
            attn_flash=False,
        ),
    )
    model = AutoregressiveWrapper(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


model     = load_model(MODEL_PATH, MAX_LENGTH, NUM_WORDS)
tokenizer = Tokenizer(MAX_LENGTH, device)

# ── Helper functions ──────────────────────────────────────────────────────────

def _parse_state(s):
    return np.array(list(s), dtype=np.int32)


def _gol_step(grid):
    ns = sum(
        np.roll(np.roll(grid, dr, 0), dc, 1)
        for dr in [-1, 0, 1] for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
    ).astype(np.int32)
    return ((ns == 3) | (grid.astype(bool) & (ns == 2))).astype(np.int32)


def run_forward_pass(current_state, next_state):
    """Return attention matrices for all LAYERS and the input/output start positions."""
    state_str     = "".join(str(int(c)) for c in current_state.flatten())
    state_str_out = "".join(str(int(c)) for c in next_state.flatten())
    full_seq      = f"@PredictNextState<{state_str}> [{state_str_out}]$"
    tokens        = tokenizer.texts_to_sequences([full_seq], do_padding=False)

    i_start = full_seq.index('<') + 1
    o_start = full_seq.index('[') + 1

    with torch.no_grad():
        embedded = model.net.token_emb(tokens)
        _, hiddens = model.net.attn_layers(embedded, return_hiddens=True)

    # list of (n_heads, seq_len, seq_len) arrays, one per requested layer
    attn_layers = [
        hiddens.attn_intermediates[l].post_softmax_attn[0].detach().cpu().numpy()
        for l in LAYERS
    ]
    del hiddens
    return attn_layers, i_start, o_start


# ── Build examples list ───────────────────────────────────────────────────────

if EXAMPLE_MODE == "all_alive":
    current_state = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    examples      = [(current_state, _gol_step(current_state))]
    example_tag   = "all_alive"
    print(f"\nUsing hardcoded all-alive board (1 example)")

elif EXAMPLE_MODE == "dataset":
    print(f"\nLoading {N_SEARCH} rows from dataset ...")
    df = pd.read_csv(DATA_CSV, nrows=N_SEARCH, dtype=str)

    # Sort by density and pick N_EXAMPLES evenly-spaced indices so that
    # the sample covers the full entropy range rather than clustering at one end.
    all_states = [(_parse_state(row["State 1"]).reshape(GRID_SIZE, GRID_SIZE),
                   _parse_state(row["State 2"]).reshape(GRID_SIZE, GRID_SIZE))
                  for _, row in df.iterrows()]
    densities  = np.array([cs.mean() for cs, _ in all_states])
    sorted_idx = np.argsort(densities)
    pick_idx   = np.linspace(0, len(sorted_idx) - 1, N_EXAMPLES, dtype=int)
    examples   = [all_states[sorted_idx[i]] for i in pick_idx]

    example_tag = f"dataset_N{len(examples)}"
    print(f"Using {len(examples)} examples spread across "
          f"densities {densities[sorted_idx[pick_idx]].min():.2f}–"
          f"{densities[sorted_idx[pick_idx]].max():.2f}")

else:
    raise ValueError(f"Unknown EXAMPLE_MODE: {EXAMPLE_MODE!r}")

# ── Accumulate neighbourhood attention scores across examples ─────────────────

score_accum = None   # (n_layers, n_heads, G, G), filled after first example
n_heads  = None
n_layers = len(LAYERS)
G        = GRID_SIZE
n_cells  = G * G

for ex_idx, (current_state, next_state) in enumerate(examples):
    print(f"  Example {ex_idx + 1}/{len(examples)} ...", end="\r")

    attn_layers, input_start, output_start = run_forward_pass(current_state, next_state)
    input_end  = input_start  + n_cells
    output_end = output_start + n_cells

    if n_heads is None:
        n_heads     = attn_layers[0].shape[0]
        score_accum = np.zeros((n_layers, n_heads, G, G), dtype=np.float64)

        # build toroidal Moore-neighbour mask once (same for all examples/layers)
        rr, cc = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        i_flat        = (rr * G + cc).flatten()
        neighbor_mask = np.zeros((n_cells, n_cells), dtype=bool)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                j_flat = ((rr + dr) % G * G + (cc + dc) % G).flatten()
                neighbor_mask[i_flat, j_flat] = True

    for li, attn_layer in enumerate(attn_layers):
        attn_to_input  = attn_layer[:, output_start:output_end, input_start:input_end]
        neighbour_attn = (attn_to_input * neighbor_mask[np.newaxis]).sum(axis=2)
        total_attn     = attn_to_input.sum(axis=2)

        with np.errstate(invalid="ignore", divide="ignore"):
            score_flat = np.where(total_attn > 1e-12, neighbour_attn / total_attn, 0.0)

        score_accum[li] += score_flat.reshape(n_heads, G, G)

print()
score_map = score_accum / len(examples)   # (n_layers, n_heads, G, G) — averaged

print(f"{n_layers} layers × {n_heads} heads, scores averaged over {len(examples)} examples")

# ── Cell-type masks ───────────────────────────────────────────────────────────

corner_mask = np.zeros((G, G), dtype=bool)
corner_mask[[0, 0, -1, -1], [0, -1, 0, -1]] = True
edge_mask   = np.zeros((G, G), dtype=bool)
edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
edge_mask   = edge_mask & ~corner_mask
bulk_mask   = ~(corner_mask | edge_mask)

# ── Plot: per-layer per-head heatmaps ─────────────────────────────────────────

print()
for li, layer_idx in enumerate(LAYERS):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Layer {layer_idx} — Moore-neighbourhood attention score  [{example_tag}]\n"
        f"fraction of input-region attention on the 8 correct (toroidal) neighbours"
        f"  —  averaged over {len(examples)} example(s)\n"
        f"1.0 = perfect  |  0.0 = attends to wrong positions",
        fontsize=10,
    )
    for h in range(n_heads):
        ax = axes[h // 4, h % 4]
        im = ax.imshow(score_map[li, h], cmap="RdYlGn", vmin=0, vmax=1,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Head {h}  (mean={score_map[li, h].mean():.2f})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"attn_nbr_layer{layer_idx:02d}_{example_tag}_heads.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Layer {layer_idx:2d}: saved {os.path.basename(out_path)}")

# ── Plot: layer × head summary heatmap (mean score per cell type) ─────────────
#
# Three side-by-side heatmaps: bulk / edge / corner mean score,
# axes are layer (rows) × head (columns).

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Neighbourhood attention score by layer × head  [{example_tag}]\n"
    f"averaged over {len(examples)} examples  |  1.0 = perfect neighbourhood routing",
    fontsize=10,
)

for ax, mask, label in zip(axes,
                            [bulk_mask, edge_mask, corner_mask],
                            ["Bulk", "Edge", "Corner"]):
    # mean_scores[li, h] = mean score over cells of this type
    data = np.array([[score_map[li, h][mask].mean()
                      for h in range(n_heads)]
                     for li in range(n_layers)])   # (n_layers, n_heads)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{LAYERS[li]}" for li in range(n_layers)], fontsize=7)

    for li in range(n_layers):
        for h in range(n_heads):
            ax.text(h, li, f"{data[li, h]:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if data[li, h] > 0.5 else "white")

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, f"attn_nbr_summary_{example_tag}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {os.path.basename(out_path)}")

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n── Neighbourhood attention score by layer / head / cell type ────────────")
header = f"{'':6}  " + "  ".join(
    f"{'L'+str(LAYERS[li]):>18}" for li in range(n_layers)
)
subhdr = f"{'':6}  " + "  ".join(
    f"{'Bulk':>5} {'Edge':>5} {'Cor':>5}" for _ in range(n_layers)
)
print(f"       " + "  ".join(f"{'Layer '+str(LAYERS[li]):^17}" for li in range(n_layers)))
print(f"{'Head':>6}  " + "  ".join(f"{'Bulk':>5} {'Edge':>5} {'Cor':>5}" for _ in range(n_layers)))
print("─" * (8 + n_layers * 20))
for h in range(n_heads):
    row = f"  H{h}    "
    for li in range(n_layers):
        b = score_map[li, h][bulk_mask].mean()
        e = score_map[li, h][edge_mask].mean()
        c = score_map[li, h][corner_mask].mean()
        row += f"{b:5.2f} {e:5.2f} {c:5.2f}  "
    print(row)

# ── Save raw scores ───────────────────────────────────────────────────────────
#
# score_map : float64 array, shape (n_layers, n_heads, GRID_SIZE, GRID_SIZE)
# layers    : list of layer indices corresponding to axis 0 of score_map
#
# Load in another script with:
#   data = np.load("attn_nbr_scores_<tag>.npz")
#   score_map = data["score_map"]   # (n_layers, n_heads, 32, 32)
#   layers    = data["layers"]      # e.g. [0, 1, ..., 11]

out_path = os.path.join(PLOTS_DIR, f"attn_nbr_scores_{example_tag}.npz")
np.savez(out_path,
         score_map=score_map,
         layers=np.array(LAYERS))
print(f"\nSaved raw scores → {os.path.basename(out_path)}")
print(f"  score_map shape: {score_map.shape}  (n_layers, n_heads, {G}, {G})")

print("\nDone.")
