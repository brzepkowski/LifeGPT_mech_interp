"""
Linear probes on LifeGPT residual streams.

For each of N_SAMPLES examples from the dataset:
  1. Run a teacher-forced forward pass and capture the residual stream at
     every LayerNorm checkpoint (2 × depth + 1 = 25 checkpoints).
  2. Compute the 8-connected neighbor sum of the current (input) state for
     every cell — the key intermediate quantity the model must learn to
     predict the GoL next state.
  3. Train a logistic-regression probe per checkpoint to predict neighbor
     sum (0–8) from the hidden state at each output position.
  4. Report and plot probe accuracy across layers.
"""

import gc
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import typing
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ── Constants ────────────────────────────────────────────────────────────────

NUM_WORDS  = 256
MAX_LENGTH = len("@PredictNextState<> []$") + (32 * 32 * 2)   # 2071
GRID_SIZE  = 32
N_SAMPLES      = 500
TRAIN_RATIO    = 0.8   # 80 % train, 20 % test
N_TRAIN        = int(N_SAMPLES * TRAIN_RATIO)
TOTAL_ROWS     = 10000
ROW_SKIP_START = 0    # skip the near-empty all-zero rows at the very start

N_CLASSES  = 9
N_EPOCHS   = 2000
PROBE_LR   = 0.1
EVAL_EVERY = 100

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR       = os.path.join(SCRIPT_DIR, "plots")
ACTIVATIONS_DIR = os.path.join(SCRIPT_DIR, "activations")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ACTIVATIONS_DIR, exist_ok=True)

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

# ── Helper classes and functions ─────────────────────────────────────────────

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
    """8-connected neighbor count per cell on a toroidal grid. Returns int32 array 0-8."""
    return sum(
        np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
    ).astype(np.int32)


def gol_step(grid: np.ndarray) -> np.ndarray:
    """One GoL step with toroidal boundary conditions."""
    n = neighbor_sum(grid)
    return ((n == 3) | (grid.astype(bool) & (n == 2))).astype(np.int32)

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_path, max_length, num_words):
    torch.cuda.empty_cache()
    model = TransformerWrapper(
        num_tokens=num_words,
        max_seq_len=max_length,
        attn_layers=Decoder(
            dim=256, depth=12, heads=8, attn_dim_head=64,
            rotary_pos_emb=True, attn_flash=True,
        ),
    )
    model = AutoregressiveWrapper(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


model    = load_model(MODEL_PATH, MAX_LENGTH, NUM_WORDS)
tokenizer = Tokenizer(MAX_LENGTH, device)

# ── Build hook infrastructure ─────────────────────────────────────────────────

_norm_cls = (torch.nn.LayerNorm,)
try:
    from x_transformers.x_transformers import RMSNorm, ScaleNorm
    _norm_cls = (*_norm_cls, RMSNorm, ScaleNorm)
except Exception:
    pass


def extract_activations(current_state: np.ndarray, next_state: np.ndarray):
    """
    Run a teacher-forced forward pass and return:
      acts  (n_checkpoints, 1024, d_model)  – residual stream at output positions
      keys  list[str]                        – checkpoint names
    """
    state_str     = "".join(str(int(c)) for c in current_state.flatten())
    state_str_out = "".join(str(int(c)) for c in next_state.flatten())
    full_seq   = f"@PredictNextState<{state_str}> [{state_str_out}]$"
    full_tokens = tokenizer.texts_to_sequences([full_seq], do_padding=False)

    _acts: dict = {}
    _hooks = []

    def _embed_hook(module, inputs, output):
        _acts["00_embed"] = output.detach().float().cpu()

    _hooks.append(model.net.token_emb.register_forward_hook(_embed_hook))

    def _make_pre_hook(name: str):
        def hook(module, inputs):
            _acts[name] = inputs[0].detach().float().cpu()
        return hook

    for name, module in model.net.attn_layers.named_modules():
        if isinstance(module, _norm_cls):
            _hooks.append(module.register_forward_pre_hook(_make_pre_hook(name)))

    with torch.no_grad():
        _out = model.net(full_tokens)

    for h in _hooks:
        h.remove()

    keys = list(_acts.keys())
    residual_streams = torch.stack(
        [_acts[k].squeeze(0) for k in keys], dim=0
    )  # (n_checkpoints, seq_len, d_model)

    output_start = full_seq.index('[') + 1
    output_end   = output_start + GRID_SIZE * GRID_SIZE
    acts = residual_streams[:, output_start:output_end, :]  # (n_checkpoints, 1024, d_model)

    return acts, keys

# ── Load dataset ──────────────────────────────────────────────────────────────

print(f"\nLoading {N_SAMPLES} examples from {os.path.basename(DATA_CSV)} ...")
_sample_idx  = np.linspace(ROW_SKIP_START, TOTAL_ROWS - 1, N_SAMPLES, dtype=int)
_rows_to_skip = sorted(set(range(1, TOTAL_ROWS + 1)) - set(_sample_idx + 1))
df = pd.read_csv(DATA_CSV, skiprows=_rows_to_skip, dtype=str)

# ── Extract activations and neighbor sums for all examples ────────────────────

all_acts:      list = []   # (N_SAMPLES,) of (n_checkpoints, 1024, d_model)
all_neighbors: list = []   # (N_SAMPLES,) of (1024,) int arrays
checkpoint_keys = None

for i, row in df.iterrows():
    current_state = np.array(list(row["State 1"]), dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
    next_state    = np.array(list(row["State 2"]), dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)

    acts, keys = extract_activations(current_state, next_state)
    all_acts.append(acts)
    all_neighbors.append(neighbor_sum(current_state).flatten())

    if checkpoint_keys is None:
        checkpoint_keys = keys

    if (i + 1) % 2 == 0 or (i + 1) == N_SAMPLES:
        print(f"  {i + 1}/{N_SAMPLES}  acts shape: {tuple(acts.shape)}")

# Example B grid (needed before we free the model)
simple_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
for _r, _c in [(0,1),(1,2),(2,0),(2,1),(2,2)]:
    simple_grid[_r + 2, _c + 2] = 1
for _r, _c in [(0,0),(0,1),(0,2)]:
    simple_grid[_r + 15, _c + 14] = 1
for _r, _c in [(0,0),(0,1),(1,0),(1,1)]:
    simple_grid[_r + 25, _c + 25] = 1
simple_next       = gol_step(simple_grid)
simple_acts, _    = extract_activations(simple_grid, simple_next)
simple_actual_grid = neighbor_sum(simple_grid).reshape(GRID_SIZE, GRID_SIZE)

# Stack → (N_SAMPLES, n_checkpoints, 1024, d_model)
all_acts_np      = torch.stack(all_acts, dim=0).numpy()
all_neighbors_np = np.stack(all_neighbors, axis=0)          # (N_SAMPLES, 1024)

n_checkpoints = all_acts_np.shape[1]
d_model       = all_acts_np.shape[3]

# Free the model from VRAM — activations are fully extracted
del model
gc.collect()
torch.cuda.empty_cache()

print(f"\nall_acts shape      : {all_acts_np.shape}")
print(f"all_neighbors shape : {all_neighbors_np.shape}")
print(f"Neighbor sum range  : {all_neighbors_np.min()} – {all_neighbors_np.max()}")

# ── Train / test split at example level ──────────────────────────────────────
# Splitting at the grid level prevents positions from the same grid leaking
# into both train and test sets (positions in the same grid are spatially correlated).

N_TEST  = N_SAMPLES - N_TRAIN
y_train = all_neighbors_np[:N_TRAIN].reshape(-1)   # for chance-accuracy print only

print(f"\nTrain : {N_TRAIN} grids  →  {N_TRAIN * GRID_SIZE * GRID_SIZE} positions")
print(f"Test  : {N_TEST}  grids  →  {N_TEST  * GRID_SIZE * GRID_SIZE} positions")
print(f"Chance accuracy (majority class): "
      f"{np.bincount(y_train).max() / len(y_train):.3f}")

# ── Train all probes in parallel on GPU ───────────────────────────────────────

n_train_pos = N_TRAIN * GRID_SIZE * GRID_SIZE
n_test_pos  = N_TEST  * GRID_SIZE * GRID_SIZE

# Reshape: (N_SAMPLES, K, 1024, D) → (K, N_SAMPLES*1024, D)
X_all_t = (torch.from_numpy(all_acts_np)
             .float()
             .permute(1, 0, 2, 3)
             .reshape(n_checkpoints, -1, d_model))
y_flat = torch.from_numpy(all_neighbors_np.reshape(-1)).long()

Xk_tr = X_all_t[:, :n_train_pos, :]
Xk_te = X_all_t[:, n_train_pos:, :]
y_tr  = y_flat[:n_train_pos]
y_te  = y_flat[n_train_pos:]

# Per-checkpoint z-score normalisation
mean_tr = Xk_tr.mean(dim=1, keepdim=True)
std_tr  = Xk_tr.std(dim=1, keepdim=True).clamp(min=1e-8)
Xk_tr   = (Xk_tr - mean_tr) / std_tr
Xk_te   = (Xk_te - mean_tr) / std_tr

Xk_tr_d  = Xk_tr.to(device)
Xk_te_d  = Xk_te.to(device)
y_tr_d   = y_tr.to(device)
y_te_d   = y_te.to(device)
y_labels = y_tr_d.repeat(n_checkpoints)   # (K * n_train_pos,)

# One weight matrix per checkpoint, trained simultaneously via bmm
W = torch.nn.init.xavier_uniform_(
    torch.empty(n_checkpoints, d_model, N_CLASSES, device=device)
).requires_grad_(True)
b = torch.zeros(n_checkpoints, N_CLASSES, device=device, requires_grad=True)

optimizer = torch.optim.Adam([W, b], lr=PROBE_LR,
                             weight_decay=1.0 / n_train_pos)
loss_fn = torch.nn.CrossEntropyLoss()

# Per-probe best-weight tracking
best_W   = W.detach().clone()
best_b   = b.detach().clone()
best_acc = torch.zeros(n_checkpoints, device=device)

print(f"\nTraining {n_checkpoints} probes in parallel on {device} ...")
for epoch in range(N_EPOCHS):
    logits = torch.bmm(Xk_tr_d, W) + b.unsqueeze(1)
    loss   = loss_fn(logits.reshape(-1, N_CLASSES), y_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % EVAL_EVERY == 0:
        with torch.no_grad():
            te_logits = torch.bmm(Xk_te_d, W) + b.unsqueeze(1)
            accs      = (te_logits.argmax(-1) == y_te_d).float().mean(dim=1)
            improved  = accs > best_acc
            best_W[improved] = W.detach()[improved]
            best_b[improved] = b.detach()[improved]
            best_acc = torch.where(improved, accs, best_acc)
        print(f"  epoch {epoch+1}/{N_EPOCHS}  loss={loss.item():.4f}  "
              f"best_acc min={best_acc.min():.3f} max={best_acc.max():.3f}")

print("  done")

with torch.no_grad():
    tr_preds = (torch.bmm(Xk_tr_d, best_W) + best_b.unsqueeze(1)).argmax(-1).cpu()
    te_preds = (torch.bmm(Xk_te_d, best_W) + best_b.unsqueeze(1)).argmax(-1).cpu()

y_tr_np = y_tr.numpy()
y_te_np = y_te.numpy()

results = []
for k, key in enumerate(checkpoint_keys):
    tr_acc  = accuracy_score(y_tr_np, tr_preds[k].numpy())
    te_acc  = accuracy_score(y_te_np, te_preds[k].numpy())
    te_bacc = balanced_accuracy_score(y_te_np, te_preds[k].numpy())
    results.append({
        "checkpoint": k,
        "key":        key,
        "train_acc":  tr_acc,
        "test_acc":   te_acc,
        "test_bacc":  te_bacc,
        "W":    best_W[k].cpu(),   # (D, C) — best weights for this probe
        "b":    best_b[k].cpu(),   # (C,)
        "mean": mean_tr[k, 0].cpu(),   # (D,)
        "std":  std_tr[k, 0].cpu(),    # (D,)
    })
    print(f"  [{k:2d}] {key:35s}  "
          f"train={tr_acc:.3f}  test={te_acc:.3f}  "
          f"balanced_test={te_bacc:.3f}")

# ── Results table ─────────────────────────────────────────────────────────────

print("\n── Probe accuracy by layer ────────────────────────────────────────────────")
print(f"{'#':>3}  {'checkpoint':<35}  {'train':>7}  {'test':>7}  {'balanced':>8}")
print("─" * 65)
for r in results:
    print(f"{r['checkpoint']:>3}  {r['key']:<35}  "
          f"{r['train_acc']:>7.3f}  {r['test_acc']:>7.3f}  {r['test_bacc']:>8.3f}")

best = max(results, key=lambda r: r["test_acc"])
print(f"\nBest probe: checkpoint {best['checkpoint']} ({best['key']})  "
      f"test accuracy = {best['test_acc']:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────

indices    = [r["checkpoint"] for r in results]
test_accs  = [r["test_acc"]   for r in results]
train_accs = [r["train_acc"]  for r in results]
test_baccs = [r["test_bacc"]  for r in results]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(indices, train_accs, marker="o", linewidth=1.5, markersize=4,
        label="Train accuracy",     color="steelblue")
ax.plot(indices, test_accs,  marker="s", linewidth=1.5, markersize=4,
        label="Test accuracy",      color="darkorange")
ax.plot(indices, test_baccs, marker="^", linewidth=1.5, markersize=4,
        label="Test balanced acc.", color="seagreen", linestyle="--")

# Chance-level baseline
chance = np.bincount(y_train).max() / len(y_train)
ax.axhline(chance, color="red", linestyle=":", linewidth=1.2, label=f"Chance ({chance:.3f})")

ax.set_xlabel("Layer checkpoint index")
ax.set_ylabel("Accuracy")
ax.set_title("Linear probe accuracy: predicting neighbor sum from residual stream")
ax.set_xticks(indices)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "probe_accuracy_by_layer.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_path}")

# ── Plot 1: test accuracy by depth (clean, dedicated) ────────────────────────

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(indices, test_accs, marker="s", linewidth=2, markersize=5, color="darkorange",
        label="Test accuracy")
ax.axhline(chance, color="red", linestyle=":", linewidth=1.2,
           label=f"Chance ({chance:.3f})")
ax.annotate(
    f"best: ckpt {best['checkpoint']}\n{best['key']}",
    xy=(best["checkpoint"], best["test_acc"]),
    xytext=(best["checkpoint"] - 4, best["test_acc"] - 0.12),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=8,
)
ax.set_xlabel("Layer checkpoint index")
ax.set_ylabel("Test accuracy")
ax.set_title("Linear probe test accuracy: predicting neighbor sum by layer")
ax.set_xticks(indices)
ax.set_xticklabels(checkpoint_keys, rotation=90, fontsize=7)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "probe_test_accuracy_by_layer.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ── Plot 2: actual vs predicted neighbor sum — two examples ───────────────────

def probe_predict(r: dict, X: np.ndarray) -> np.ndarray:
    """Predict neighbor sum using a stored probe. X: (n, d_model) float32 array."""
    Xt = (torch.from_numpy(X.astype(np.float32)) - r["mean"]) / r["std"]
    return (Xt @ r["W"] + r["b"]).argmax(dim=-1).numpy()

best_r = results[best["checkpoint"]]

# Example A: high-entropy grid from the test split
test_grid   = np.array(list(df.iloc[N_TRAIN]["State 1"]), dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
test_acts   = all_acts_np[N_TRAIN, best["checkpoint"], :, :]
test_pred   = probe_predict(best_r, test_acts).reshape(GRID_SIZE, GRID_SIZE)
test_actual = all_neighbors_np[N_TRAIN].reshape(GRID_SIZE, GRID_SIZE)
test_err    = (test_pred != test_actual).astype(int)

# Example B: sparse, human-readable pattern (glider + blinker + 2×2 block)
# (simple_grid, simple_acts, simple_actual_grid extracted before model was freed)
simple_actual = simple_actual_grid
simple_pred   = probe_predict(
                    best_r, simple_acts[best["checkpoint"]].numpy()
                ).reshape(GRID_SIZE, GRID_SIZE)
simple_err    = (simple_pred != simple_actual).astype(int)

# Draw both rows
examples = [
    ("High-entropy grid", test_grid,    test_actual,   test_pred,   test_err),
    ("Sparse pattern (glider + blinker + block)", simple_grid, simple_actual, simple_pred, simple_err),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
cmap, vmin, vmax = "YlOrRd", 0, 8

for row, (label, grid, actual, pred, err) in enumerate(examples):
    im0 = axes[row, 0].imshow(grid, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    axes[row, 0].set_title(f"{label}\n({grid.sum()} live cells)", fontsize=11)
    axes[row, 0].axis("off")
    plt.colorbar(im0, ax=axes[row, 0], ticks=[0, 1])

    im1 = axes[row, 1].imshow(actual, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[row, 1].set_title("Actual neighbor sum", fontsize=11)
    axes[row, 1].axis("off")
    plt.colorbar(im1, ax=axes[row, 1], ticks=range(9))

    im2 = axes[row, 2].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[row, 2].set_title(
        f"Predicted (ckpt {best['checkpoint']})\nacc = {best['test_acc']:.3f}", fontsize=11)
    axes[row, 2].axis("off")
    plt.colorbar(im2, ax=axes[row, 2], ticks=range(9))

    im3 = axes[row, 3].imshow(err, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[row, 3].set_title(
        f"Errors  ({err.sum()} / {GRID_SIZE * GRID_SIZE} cells wrong)", fontsize=11)
    axes[row, 3].axis("off")
    plt.colorbar(im3, ax=axes[row, 3])

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "best_probe_prediction_vs_actual.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ── Save probe results ────────────────────────────────────────────────────────

save_path = os.path.join(ACTIVATIONS_DIR, "probe_results.pt")
torch.save({
    "results":         results,          # list of dicts with W/b/mean/std per checkpoint
    "checkpoint_keys": checkpoint_keys,
    "all_acts":        all_acts_np,
    "all_neighbors":   all_neighbors_np,
    "N_TRAIN":         N_TRAIN,
    "N_TEST":          N_TEST,
    "grid_size":       GRID_SIZE,
}, save_path, pickle_protocol=4)
print(f"Saved probe results to {save_path}")
