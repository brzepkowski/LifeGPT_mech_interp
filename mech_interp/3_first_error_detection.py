"""
3_first_error_detection.py

Runs the ARAR (autoregressive autoregressor) loop for three initial conditions
that were shown to produce errors in the paper:
  - Order Param: 0.25   (row index 1 in the test CSV)
  - Glider              (row index 5)
  - r-Pentomino         (row index 9)

For each IC, the model is run autoregressively (output fed back as next input)
and every generated state is compared against the ground-truth state stored in
the test CSV.  The loop stops as soon as the first divergence is found and
prints the step number, error count, error rate, and positions of wrong cells.
A summary plot is saved to mech_interp/plots/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import typing
import pandas as pd

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TEST_CSV = os.path.join(
    REPO_ROOT, "LifeGPT",
    "conway_test_states_32by32_20240827_172807.csv",
)
MODEL_PATH = os.path.join(
    REPO_ROOT,
    "model_parameters",
    "07_22_2024_Conway_2_State_Jump_Rot_Pos_On_Masking_On_Broad_Entropy_Homog_2024-07-23 10-37-31",
    "LifeGPT_epoch_50.pt",
)

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_WORDS       = 256
GRID_SIZE       = 32
MAX_LENGTH      = len("@PredictNextState<> []$") + (GRID_SIZE * GRID_SIZE * 2)  # 2071
GENERATE_LENGTH = MAX_LENGTH - len("@PredictNextState<>") - (GRID_SIZE * GRID_SIZE)  # 1028
MAX_ARAR_STEPS  = 249  # match the notebook

# Row indices (0-based) in the test CSV for the three target ICs
TARGET_CASES = {
    1: "Order Param: 0.25",
    5: "Glider",
    9: "r-Pentomino",
}

# ── Tokenizer ─────────────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, n_pad: int, device: torch.device, pad_byte: int = 0):
        self.n_pad   = n_pad
        self.device  = device
        self.pad_byte = pad_byte

    def tokenize_str(self, sentence: str, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert len(base) == self.n_pad
        return torch.tensor(base, dtype=torch.long, device=self.device)

    def texts_to_sequences(self, texts: typing.List[str], encoding="utf8", do_padding=True):
        seqs = [self.tokenize_str(t, encoding, do_padding).unsqueeze(0) for t in texts]
        return torch.cat(seqs, dim=0).to(self.device)

    def sequences_to_texts(self, texts: torch.Tensor, encoding="utf8"):
        out = []
        for seq in texts:
            chars = [int(seq[i]) for i in range(len(seq)) if seq[i] != 0]
            try:
                out.append(bytes(chars).decode(encoding))
            except Exception:
                pass
        return out


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device) -> AutoregressiveWrapper:
    torch.cuda.empty_cache()
    model = TransformerWrapper(
        num_tokens=NUM_WORDS,
        max_seq_len=MAX_LENGTH,
        attn_layers=Decoder(
            dim=256, depth=12, heads=8, attn_dim_head=64,
            rotary_pos_emb=True, attn_flash=True,
        ),
    )
    model = AutoregressiveWrapper(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def extract_task(s: str, end_task_token: str = ">") -> str:
    return s[: s.find(end_task_token) + 1]


def extract_sample(s: str, start_token: str = "[", end_token: str = "]") -> str:
    return s[s.find(start_token) + 1: s.find(end_token)]


# ── ARAR loop with early stopping on first error ──────────────────────────────

def find_first_error(
    model: AutoregressiveWrapper,
    tokenizer: Tokenizer,
    device: torch.device,
    initial_state_str: str,
    ground_truth_states: list,
) -> dict:
    """
    Runs the ARAR loop, comparing each generated state to the corresponding
    ground-truth state.  Returns a dict with results for the first diverging step.

    ground_truth_states[i] is the ground-truth string for step i+1
    (i.e. index 0 = State 2, index 1 = State 3, ...).
    """
    current_input = extract_task(
        f"@PredictNextState<{initial_state_str}>", end_task_token=">"
    )

    for step in range(MAX_ARAR_STEPS):
        # Pass the prompt string directly (not as a list); texts_to_sequences
        # iterates over its characters giving shape (seq_len, 1), then the
        # transpose produces the correct (1, seq_len) batch for model.generate.
        inp = torch.Tensor(
            tokenizer.texts_to_sequences(current_input, do_padding=False)
        ).to(device).transpose(0, 1).long()

        with torch.no_grad():
            sample = model.generate(
                prompts=inp,
                seq_len=GENERATE_LENGTH,
                temperature=0,
                cache_kv=True,
            )

        decoded = tokenizer.sequences_to_texts(sample[:1])
        generated = extract_sample(decoded[0]) if decoded else ""

        gt = ground_truth_states[step]

        # Pad/trim to 1024 just in case
        generated_cmp = generated.ljust(GRID_SIZE * GRID_SIZE, "0")[: GRID_SIZE * GRID_SIZE]
        gt_cmp        = str(gt).ljust(GRID_SIZE * GRID_SIZE, "0")[: GRID_SIZE * GRID_SIZE]

        wrong_positions = [k for k in range(len(gt_cmp)) if generated_cmp[k] != gt_cmp[k]]
        n_wrong         = len(wrong_positions)

        del sample, inp
        torch.cuda.empty_cache()

        if n_wrong > 0:
            return {
                "first_error_step": step + 2,  # step+2 because step 0 predicts State 2
                "n_wrong_cells": n_wrong,
                "error_rate": n_wrong / (GRID_SIZE * GRID_SIZE),
                "wrong_positions": wrong_positions,
                "generated": generated_cmp,
                "ground_truth": gt_cmp,
                "arar_step_index": step,
            }

        # Feed the generated state back as next input
        current_input = f"@PredictNextState<{generated}>"
        print(f"  step {step + 2:>3d}: perfect match")

    return {"first_error_step": None}


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_first_error(case_name: str, result: dict, initial_state_str: str):
    if result["first_error_step"] is None:
        print(f"  [{case_name}] No errors found in {MAX_ARAR_STEPS} steps.")
        return

    step      = result["first_error_step"]
    gen_str   = result["generated"]
    gt_str    = result["ground_truth"]

    gen_grid  = np.array([int(c) for c in gen_str], dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
    gt_grid   = np.array([int(c) for c in gt_str],  dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
    init_grid = np.array([int(c) for c in initial_state_str], dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)

    diff = gen_grid.astype(int) - gt_grid.astype(int)
    diff_rgb          = np.ones((*diff.shape, 3))
    diff_rgb[diff ==  1] = [1.0, 0.2, 0.2]  # false positive → red
    diff_rgb[diff == -1] = [0.2, 0.4, 1.0]  # false negative → blue

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(init_grid, cmap="binary", vmin=0, vmax=1)
    axes[0].set_title("IC (State 1)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gt_grid, cmap="binary", vmin=0, vmax=1)
    axes[1].set_title(f"Ground truth\n(State {step})", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(gen_grid, cmap="binary", vmin=0, vmax=1)
    axes[2].set_title(f"Model prediction\n(State {step})", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(diff_rgb)
    axes[3].set_title(
        f"First error map\n{result['n_wrong_cells']} wrong cells "
        f"({result['error_rate']:.3f})",
        fontsize=11,
    )
    axes[3].axis("off")

    safe_name = case_name.replace(":", "").replace(" ", "_").replace(".", "p")
    fig.suptitle(f"{case_name}  —  first error at State {step}", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"first_error_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(TEST_CSV)
    print(f"Test CSV loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    model     = load_model(MODEL_PATH, device)
    tokenizer = Tokenizer(MAX_LENGTH, device)

    summary = []

    for row_idx, case_name in TARGET_CASES.items():
        print(f"\n{'='*60}")
        print(f"Case: {case_name}  (row index {row_idx})")
        print("="*60)

        # State 1 is the initial condition; States 2..250 are ground truth targets
        initial_state_str  = str(df.iloc[row_idx]["State 1"])
        ground_truth_states = [str(df.iloc[row_idx][f"State {s}"]) for s in range(2, 251)]

        result = find_first_error(
            model, tokenizer, device,
            initial_state_str, ground_truth_states,
        )

        if result["first_error_step"] is None:
            print(f"  No errors in {MAX_ARAR_STEPS} ARAR steps — perfect run.")
        else:
            s   = result["first_error_step"]
            n   = result["n_wrong_cells"]
            er  = result["error_rate"]
            pos = result["wrong_positions"]
            rows_cols = [(p // GRID_SIZE, p % GRID_SIZE) for p in pos]
            print(f"  First error at State {s}")
            print(f"  Wrong cells : {n}  (error rate {er:.4f})")
            print(f"  Cell positions (row, col): {rows_cols[:10]}"
                  + (" ..." if len(pos) > 10 else ""))

        plot_first_error(case_name, result, initial_state_str)
        summary.append((case_name, result))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Case':<25}  {'First error at State':>20}  {'Wrong cells':>11}  {'Error rate':>10}")
    print("-"*72)
    for name, res in summary:
        if res["first_error_step"] is None:
            print(f"{name:<25}  {'No errors':>20}  {'—':>11}  {'—':>10}")
        else:
            print(f"{name:<25}  {res['first_error_step']:>20}  "
                  f"{res['n_wrong_cells']:>11}  {res['error_rate']:>10.4f}")


if __name__ == "__main__":
    main()
