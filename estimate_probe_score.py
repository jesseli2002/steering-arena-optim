# %%
import argparse


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactively estimate the mean Season 2 probe score for "
        "steering prompts, loading the model once and looping over stdin."
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test on smaller model (GPT-2), run locally to test pipeline",
    )
    return ap.parse_args(argv)


args = parse_args()

# %%
import json
import os
from pathlib import Path
import gc

import numpy as np
import torch as t
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    load_direction,
    load_prompt_suffixes,
    truncate_to_layer,
    compute_scores_batch,
)

# %%
t.manual_seed(505080078)

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
ARENA_ROOT = REPO_ROOT / "steering-arena"

if not ARENA_ROOT.is_dir() and os.environ.get("STEERING_ARENA_DIR"):
    ARENA_ROOT = Path(os.environ["STEERING_ARENA_DIR"]).resolve()
assert (
    ARENA_ROOT.is_dir()
), f"expected read-only steering-arena clone at {ARENA_ROOT} (or set STEERING_ARENA_DIR)"


HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, f"Please set HF_TOKEN in .env"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")  # temporary override
dtype = t.bfloat16  # for training
if device.type != "cuda":
    print("\033[93mWarning: CUDA not available, using CPU. This will be slow.\033[0m")


# %%
SMOKE = args.smoke

# Load probe direction & season prompt suffix
SEASON_FILE = ARENA_ROOT / "data" / "probes" / "season2.json"
if SMOKE:
    DIRECTION_FILE = REPO_ROOT / "data_local/directions/d_dev_smoke.npz"
    REPORT_FILE = REPO_ROOT / "data_local" / "score_check.json"
else:
    DIRECTION_FILE = ARENA_ROOT / "data/directions/d_olmo3_L24_logistic.npz"
    REPORT_FILE = REPO_ROOT / "data" / "score_check.json"

TEST_PREFIXES_FILE = ARENA_ROOT / "data" / "test" / "prefixes.json"

# %%
# Device convention: Probe scoring is computed on CPU with float32.
probe_dir, meta = load_direction(DIRECTION_FILE)
probe_dir = t.tensor(
    probe_dir / np.linalg.norm(probe_dir), device="cpu", dtype=t.float32
)
suffixes = load_prompt_suffixes(SEASON_FILE)

print(f"direction: {DIRECTION_FILE}")
print(
    f"  model_id={meta.get('model_id')} layer={meta.get('layer')} placeholder={meta.get('placeholder')}"
)
print(f"suffixes: {len(suffixes)} from {SEASON_FILE}")
if meta.get("placeholder"):
    print("[warning] using a PLACEHOLDER direction -- scores are not meaningful yet.")

MODEL_NAME = meta["model_id"]
LAYER = int(meta["layer"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

max_memory = None
# max_memory = {i: "20GiB" for i in range(4)}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=dtype, device_map="auto", max_memory=max_memory, token=HF_TOKEN
)
# if model.device != device:  # in debugging, might use CPU
#     model = model.to(device)

model.requires_grad_(False)

# Only layer LAYER's residual stream is read, so drop the blocks above it and
# run the forward through the trunk (skips the final norm / lm_head). For
# Olmo-3-32B this keeps 26 of 64 layers (~2.5x faster per forward). The freed
# blocks are also released from VRAM.
truncate_to_layer(model, LAYER)
trunk = model.base_model
gc.collect()
t.cuda.empty_cache()

# sfx -> suffix tokens for the composed (ctrl + " " + suffix) forward pass.
# add_special_tokens=False: the sequence's BOS comes from ctrl_tokens below
# (which is first in the concatenation); without this, tokenizers that
# auto-prepend BOS (e.g. Llama/OLMo-style) would insert a second one here,
# mid-sequence.
sfx_enc = tokenizer(
    [" " + suffix for suffix in suffixes],
    padding=True,
    add_special_tokens=False,
    return_tensors="pt",
)
sfx_mask = sfx_enc["attention_mask"]  # (n_sfx, seq)
n_sfx_tokens = sfx_mask.sum(axis=1)  # (n_sfx,)
sfx_tokens = sfx_enc["input_ids"]  # (n_sfx, seq)

#  constant
sfx_embed = model.get_input_embeddings()(sfx_tokens.to(device))

# Baseline: cos(R(suffix alone)[-1], d) for each suffix, tokenized standalone
# (own BOS, no leading space) -- matches the server's shift score,
# score(seq) = mean_i cos(R(seq+" "+suffix_i))  - cos(R(suffix_i)).
# Mean is linear, so it's enough to compute one baseline scalar up front
# and subtract it from every candidate's already suffix-averaged score,
# rather than threading per-suffix baselines through compute_scores_batch.
base_enc = tokenizer(suffixes, padding=True, return_tensors="pt")
base_mask = base_enc["attention_mask"]
n_base_tokens = base_mask.sum(axis=1)
with t.no_grad():
    base_out = trunk(
        input_ids=base_enc["input_ids"].to(device),
        attention_mask=base_mask.to(device),
        output_hidden_states=True,
    )
    base_h = base_out.hidden_states[LAYER + 1]  # (n_sfx, seq, d_model)
    base_pos = n_base_tokens - 1
    base_acts = base_h[t.arange(len(suffixes)), base_pos].cpu().float()
    baseline_scores = (base_acts @ probe_dir) / t.linalg.norm(base_acts, dim=-1)
baseline_mean = baseline_scores.mean().item()
print(f"baseline (probe-alone) mean cosine: {baseline_mean:+.6f}")


def load_prompts_json():
    return json.loads(TEST_PREFIXES_FILE.read_text(encoding="utf-8"))


# We'll just add our scores directly to results.
# Note - expect to get slight differences due to tokenization quirks where controlled prompt gets prepended to suffix.
results = load_prompts_json()

for entry in results:
    prompt = entry["prompt"]

    ctrl_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    ctrl_embed = model.get_input_embeddings()(ctrl_tokens.to(device))

    raw_score = compute_scores_batch(
        trunk,
        ctrl_embed,
        sfx_embed,
        sfx_mask,
        n_sfx_tokens,
        probe_dir,
        layer=LAYER,
        chunk=1,
    ).item()
    score_estimate = raw_score - baseline_mean

    # Modify entry in place
    entry["score_estimate"] = score_estimate
    gt = entry.get("score")
    if gt is not None:
        entry["diff"] = score_estimate - gt
        print(
            f"  estimate={score_estimate:+.6f}  gt={gt:+.6f}  diff={score_estimate - gt:+.6f}  {prompt!r}"
        )
    else:
        print(f"  estimate={score_estimate:+.6f}  {prompt!r}")

REPORT_FILE.write_text(json.dumps(results, indent=2))
print(f"wrote {REPORT_FILE}")
