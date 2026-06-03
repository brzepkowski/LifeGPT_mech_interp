import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
import typing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ── Constants ────────────────────────────────────────────────────────────────

NUM_WORDS = 256

MAX_LENGTH = len("@PredictNextState<> []$") + (32 * 32 * 2)   # 2071
GENERATE_LENGTH = MAX_LENGTH - len("@PredictNextState<>") - (32 * 32)  # 1028

GRID_SIZE = 32

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR      = os.path.join(SCRIPT_DIR, "plots")
ACTIVATIONS_DIR = os.path.join(SCRIPT_DIR, "activations")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ACTIVATIONS_DIR, exist_ok=True)

# ── Helper classes and functions ─────────────────────────────────────────────

class Tokenizer:
    def __init__(self, n_pad: int, device: torch.device, pad_byte: int = 0):
        self.n_pad = n_pad
        self.device = device
        self.pad_byte = pad_byte

    def tokenize_str(self, sentence: str, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert len(base) == self.n_pad, f"n_pad is too small, use {len(base)} or greater."
        tensor = torch.Tensor(base)
        return tensor.long().to(self.device)

    def texts_to_sequences(self, texts: typing.List[str], encoding="utf8", do_padding=True):
        sentences = [self.tokenize_str(sentence, do_padding=do_padding).unsqueeze(0) for sentence in texts]
        return torch.cat(sentences, dim=0).to(self.device)

    def sequences_to_texts(self, texts: torch.Tensor, encoding="utf8"):
        out = []
        for seq in texts:
            chars = []
            i = 0
            while i < len(seq) and seq[i] != 0:
                chars.append(int(seq[i]))
                i += 1
            try:
                out.append(bytes(chars).decode(encoding))
            except Exception:
                pass
        return out


def empty_cuda_cache():
    torch.cuda.empty_cache()


def extract_sample(string_input, start_token='[', end_token=']'):
    i = string_input.find(start_token)
    j = string_input.find(end_token)
    return string_input[i + 1:j]


def extract_task(string_input, end_task_token='>'):
    j = string_input.find(end_task_token)
    return string_input[:j + 1]

# ── Model ────────────────────────────────────────────────────────────────────

def load_model(model_path, max_length, num_words):
    empty_cuda_cache()

    model = TransformerWrapper(
        num_tokens=num_words,
        max_seq_len=max_length,
        attn_layers=Decoder(
            dim=256,
            depth=12,
            heads=8,
            attn_dim_head=64,
            rotary_pos_emb=True,
            attn_flash=True,
        ),
    )
    model = AutoregressiveWrapper(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"Model loaded from {model_path}")
    return model


model_path = "../model_parameters/07_22_2024_Conway_2_State_Jump_Rot_Pos_On_Masking_On_Broad_Entropy_Homog_2024-07-23 10-37-31/LifeGPT_epoch_50.pt"
# model_path = "../model_parameters/08_14_2024_Conway_2_State_Jump_Rot_Pos_On_Masking_Off_High_Entropy_Homog_2024-08-14 12-24-10/LifeGPT_epoch_16.pt"
model = load_model(model_path, MAX_LENGTH, NUM_WORDS)

tokenizer = Tokenizer(MAX_LENGTH, device)

# ── Hardcoded initial state ───────────────────────────────────────────────────

grid_init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

# Glider (top-left, row-offset=2, col-offset=2)
glider_cells = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
for r, c in glider_cells:
    grid_init[r + 2, c + 2] = 1

# Blinker (bottom-right, vertical orientation)
blinker_cells = [(0, 0), (1, 0), (2, 0)]
for r, c in blinker_cells:
    grid_init[r + 26, c + 26] = 1

# R-pentomino (centre)
rpent_cells = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]
for r, c in rpent_cells:
    grid_init[r + 14, c + 14] = 1

print(f"Live cells: {grid_init.sum()}")

# ── Exact next state ──────────────────────────────────────────────────────────

def gol_step(grid: np.ndarray) -> np.ndarray:
    neighbors = sum(
        np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
    )
    return ((neighbors == 3) | (grid.astype(bool) & (neighbors == 2))).astype(np.int32)


grid_exact = gol_step(grid_init)
print(f"Live cells after one step: {grid_exact.sum()}")

# ── Plot: initial state vs. exact next state ──────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(grid_init, cmap="binary", vmin=0, vmax=1)
axes[0].set_title("Initial state", fontsize=13)
axes[0].axis("off")

axes[1].imshow(grid_exact, cmap="binary", vmin=0, vmax=1)
axes[1].set_title("Next state (exact GoL rules)", fontsize=13)
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "initial_vs_exact.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/initial_vs_exact.png")

# ── Convert initial state to model input format ───────────────────────────────

state_str = "".join(str(int(cell)) for cell in grid_init.flatten())
input_seq = f"@PredictNextState<{state_str}>"

print(f"Prompt length : {len(input_seq)} chars")
print(f"Prompt preview: {input_seq[:40]}...{input_seq[-10:]}")

# ── Run inference (sanity check) ──────────────────────────────────────────────

inp = extract_task(input_seq, end_task_token='>')
inp = torch.Tensor(tokenizer.texts_to_sequences(inp, do_padding=False)).to(device)
inp = inp.transpose(0, 1).long()
print(f"inp.shape: {inp.shape}")

with torch.no_grad():
    sample = model.generate(
        prompts=inp,
        seq_len=GENERATE_LENGTH,
        temperature=0,
        cache_kv=True,
    )

print(sample.shape)

try:
    output_str = tokenizer.sequences_to_texts(sample[:1])
    pred_str = extract_sample(output_str[0])
    print(f"Prediction: {pred_str}")
except Exception as e:
    print(f"Error decoding output: {e}")
    pred_str = None

if pred_str and len(pred_str) == GRID_SIZE * GRID_SIZE:
    grid_pred = np.array([int(b) for b in pred_str], dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
    print(f"Prediction decoded successfully. Live cells: {grid_pred.sum()}")
else:
    print(f"Decode failed or wrong length (got {len(pred_str) if pred_str else 'None'} bits).")
    grid_pred = None

# ── Plot: exact next state vs. model prediction ───────────────────────────────

if grid_pred is not None:
    n_cells   = GRID_SIZE * GRID_SIZE
    n_correct = int((grid_exact == grid_pred).sum())
    accuracy  = n_correct / n_cells
    n_wrong   = n_cells - n_correct

    diff = grid_pred.astype(int) - grid_exact.astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(grid_exact, cmap="binary", vmin=0, vmax=1)
    axes[0].set_title("Exact next state (GoL rules)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(grid_pred, cmap="binary", vmin=0, vmax=1)
    axes[1].set_title(f"Model prediction\n(accuracy = {accuracy:.4f})", fontsize=12)
    axes[1].axis("off")

    diff_rgb = np.ones((*diff.shape, 3))
    diff_rgb[diff ==  1] = [1.0, 0.2, 0.2]   # false positive → red
    diff_rgb[diff == -1] = [0.2, 0.4, 1.0]   # false negative → blue
    axes[2].imshow(diff_rgb)
    axes[2].set_title(
        f"Error map ({n_wrong} wrong cells)\n"
        f"red = false positive, blue = false negative",
        fontsize=12,
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exact_vs_prediction.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/exact_vs_prediction.png")
else:
    print("Cannot plot: model output could not be decoded.")

# ── Extract hidden states ─────────────────────────────────────────────────────
#
# Strategy: teacher-forced forward pass over the complete sequence
# (prompt + ground-truth next state).  Because the model is causal, the
# residual stream at output position j and layer L is identical to what the
# model uses during autoregressive generation to predict bit j — but we get
# all positions in a single forward pass.
#
# Capture points: inputs to every LayerNorm inside the decoder.
# In a pre-norm transformer the residual stream reaches each LayerNorm
# without modification, so these pre-hooks give clean residual-stream
# snapshots at 2 × depth checkpoints (one before attention, one before
# feed-forward, per layer).

# Build the full sequence: prompt + ' [' + 1024 output bits + ']$'
state_str_out = "".join(str(int(c)) for c in grid_exact.flatten())
full_seq = f"@PredictNextState<{state_str}> [{state_str_out}]$"
full_tokens = tokenizer.texts_to_sequences([full_seq], do_padding=False)  # (1, seq_len)

print(f"\nFull sequence length : {len(full_seq)} chars  (expected {MAX_LENGTH})")
print(f"Token tensor shape   : {full_tokens.shape}")

# Register hooks
_activations: dict = {}   # insertion order preserved (Python ≥ 3.7) = forward-pass order
_hooks = []

# (1) Token-embedding output = initial residual stream (before any attention)
def _embed_hook(module, inputs, output):
    _activations["00_embed"] = output.detach().float().cpu()

_hooks.append(model.net.token_emb.register_forward_hook(_embed_hook))

# (2) Pre-norm inputs for every LayerNorm inside the decoder
_norm_cls = (torch.nn.LayerNorm,)
try:
    from x_transformers.x_transformers import RMSNorm, ScaleNorm
    _norm_cls = (*_norm_cls, RMSNorm, ScaleNorm)
except Exception:
    pass

def _make_pre_hook(name: str):
    def hook(module, inputs):
        _activations[name] = inputs[0].detach().float().cpu()
    return hook

for name, module in model.net.attn_layers.named_modules():
    if isinstance(module, _norm_cls):
        _hooks.append(module.register_forward_pre_hook(_make_pre_hook(name)))

print(f"Registered {len(_hooks)} hooks  "
      f"({len(_hooks) - 1} LayerNorm pre-hooks + 1 embed hook)")

# Run teacher-forced forward pass
with torch.no_grad():
    _out = model.net(full_tokens)

for h in _hooks:
    h.remove()
_hooks.clear()

logits = _out[0] if isinstance(_out, tuple) else _out
print(f"Logits shape             : {tuple(logits.shape)}")
print(f"Residual-stream captures : {len(_activations)}")

# Stack → (n_checkpoints, seq_len, d_model)
checkpoint_keys = list(_activations.keys())
residual_streams = torch.stack(
    [_activations[k].squeeze(0) for k in checkpoint_keys], dim=0
)

print(f"\nResidual stream tensor : {tuple(residual_streams.shape)}")
print(f"  n_checkpoints = {residual_streams.shape[0]}")
print(f"  seq_len       = {residual_streams.shape[1]}")
print(f"  d_model       = {residual_streams.shape[2]}")

# ── Identify output positions ─────────────────────────────────────────────────
#
# Sequence layout:
#   @PredictNextState< {1024 prompt bits} > [ {1024 output bits} ] $
#                                            ^
#                                            bracket_open
# The 1024 output bits begin one position after '['.

bracket_open = full_seq.index('[')
output_start = bracket_open + 1
output_end   = output_start + GRID_SIZE * GRID_SIZE

print(f"\nSequence layout:")
print(f"  '[' token   : position {bracket_open}")
print(f"  output bits : positions {output_start} .. {output_end - 1}")

# (n_checkpoints, 1024, d_model) — residual stream at each output bit, all layers
acts = residual_streams[:, output_start:output_end, :]

# (1024,) — ground-truth next-state labels (0 = dead, 1 = alive)
labels = torch.tensor(grid_exact.flatten(), dtype=torch.long)

print(f"\nacts   shape : {tuple(acts.shape)}")
print(f"labels shape : {tuple(labels.shape)}  (live cells: {labels.sum().item()})")

# ── Save activations ──────────────────────────────────────────────────────────

save_path = os.path.join(ACTIVATIONS_DIR, "residual_streams_glider_blinker_rpent.pt")
torch.save({
    "residual_streams": residual_streams,   # (n_checkpoints, seq_len, d_model)
    "acts":             acts,               # (n_checkpoints, 1024, d_model)
    "labels":           labels,             # (1024,)  0/1 ground-truth next-state bits
    "checkpoint_keys":  checkpoint_keys,    # list[str] — maps index → layer name
    "full_seq":         full_seq,
    "output_start":     output_start,
    "grid_init":        torch.tensor(grid_init),
    "grid_exact":       torch.tensor(grid_exact),
    "grid_size":        GRID_SIZE,
}, save_path)

print(f"\nSaved activations to {save_path}")
print(f"  residual_streams : {tuple(residual_streams.shape)}")
print(f"  acts             : {tuple(acts.shape)}")
print(f"  labels           : {tuple(labels.shape)}")
