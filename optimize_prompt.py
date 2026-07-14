# %%
"""
Optimize the prompt using "Greedy Coordinate Gradients"
"""

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
import hashlib
import json
import os
from pathlib import Path
import gc
import time
import datetime as dt

import numpy as np
import torch as t
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    cos_sim,
    compose,
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
else:
    DIRECTION_FILE = ARENA_ROOT / "data/directions/d_olmo3_L24_logistic.npz"

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

CACHE_DIR = REPO_ROOT / ".cache" / "acts" / MODEL_NAME.replace("/", "_")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

max_memory = None
# max_memory = {i: "20GiB" for i in range(4)}
max_memory = {0: "15GiB", 1: "17GiB", 2: "20GiB", 3: "20GiB"}
print(f"{max_memory=}")

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

# %%
if SMOKE:
    NUM_LAYERS = model.config.n_layer
    D_MODEL = model.config.n_embd
    D_VOCAB = model.config.vocab_size
else:
    NUM_LAYERS = model.config.num_hidden_layers
    D_MODEL = model.config.hidden_size
    D_VOCAB = model.config.vocab_size

# Some algorithm hyperparameters
N_CONTROLLED_TOKENS = 24

# Candidates scored per forward pass. Main memory/speed knob: higher packs more
# candidates into a single batched forward (faster) but uses more activation
# memory. Each unit of chunk costs one current-prefix-sized forward.
CAND_CHUNK = 3

if SMOKE:  # for debugging
    BATCH_SIZE_OPTIM = 32

# Checkpointing / experiment tracking
RESUME_FROM = None  # None = fresh run; else path to an existing run dir to continue
# RESUME_FROM = "data/optimization/2026-07-13T14-11-04Z"  # None = fresh run; else path to an existing run dir to continue
USE_WANDB = True  # attempts wandb; degrades to a no-op if import/init/log fails

# %%
# sfx -> suffix tokens
sfx_enc = tokenizer(
    [" " + suffix for suffix in suffixes],
    padding=True,
    return_tensors="pt",
    add_special_tokens=False,
)
n_sfx_tokens = sfx_enc["attention_mask"].sum(axis=1)  # (batch, )
sfx_tokens = sfx_enc["input_ids"]  # (batch, seq)

#  constant
sfx_embed = model.get_input_embeddings()(sfx_tokens.to(device))

# Initialization of optimized value. Random start/restart might do better, but that's for later.
ctrl_token_ids = t.full(
    (N_CONTROLLED_TOKENS,), tokenizer("!")["input_ids"][0], device=device
)

# %%
# Optimization loop


def compute_score_gradient(ctrl_token_ids):
    """Current mean probe score and its gradient w.r.t. the control tokens.

    Builds a one-hot encoding of the control prefix, scores it through compute_scores_batch, and backprops to get the gradient of the score w.r.t. that one-hot (the GCG signal).

    :param ctrl_token_ids: (N_CONTROLLED_TOKENS,) current prompt-prefix tokens
    :returns: (score, grad) where score is the float mean probe score and grad is the (N_CONTROLLED_TOKENS, D_VOCAB) gradient of the score w.r.t. the control-token one-hot.
    """
    # Define controlled tokens
    ctrl_tokens_onehot = t.zeros(  # (seq, d_vocab)
        (N_CONTROLLED_TOKENS, D_VOCAB), dtype=dtype, device=device
    )
    ctrl_tokens_onehot[t.arange(N_CONTROLLED_TOKENS), ctrl_token_ids] = 1
    ctrl_tokens_onehot.requires_grad = True

    embed_weights = model.get_input_embeddings().weight  # (d_vocab, d_model)
    ctrl_embed = (ctrl_tokens_onehot @ embed_weights)[None]  # (1, seq, d_model)

    # TODO: Check with steering-arena author their layer numbering convention for probe.
    scores = compute_scores_batch(
        trunk,
        ctrl_embed,
        sfx_embed,
        sfx_enc["attention_mask"],
        n_sfx_tokens,
        probe_dir,
        LAYER,
    )
    mean_score = scores[0]
    mean_score.backward()  # Maximize score => pick top k gradients
    return mean_score.item(), ctrl_tokens_onehot.grad


# %%
# Checkpointing and experiment tracking setup

ARTEFACT_ROOT = REPO_ROOT / ("data_local" if SMOKE else "data")


def decode_prompt(ids):
    """Decode a 1-D tensor of control token ids to the prompt prefix string."""
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def save_json_atomic(path: Path, obj):
    """Write JSON to `path` atomically via a temp file + replace."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


if RESUME_FROM is not None:
    run_dir = Path(RESUME_FROM)
    assert run_dir.is_dir(), f"RESUME_FROM is not a directory: {run_dir}"
    run_id = run_dir.name
else:
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = ARTEFACT_ROOT / "optimization" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

LATEST = run_dir / "latest.json"
BEST = run_dir / "best.json"
HISTORY = run_dir / "history.jsonl"
print(f"run dir: {run_dir}")

# WandB is optional; any failure (missing package, offline) degrades to a no-op.
wandb_run = None
if USE_WANDB:
    try:
        import wandb

        wandb_run = wandb.init(
            project="steering-arena-optim",
            config={
                "N_CONTROLLED_TOKENS": N_CONTROLLED_TOKENS,
                "model_id": MODEL_NAME,
                "layer": LAYER,
                "smoke": SMOKE,
                "run_id": run_id,
            },
        )
    except Exception as e:
        print(f"\033[93m[wandb] disabled ({e})\033[0m")
        wandb_run = None


def wandb_log(d):
    if wandb_run is not None:
        try:
            wandb_run.log(d)
        except Exception:
            pass


# Resume from checkpoint or start fresh
if RESUME_FROM is not None:
    state = json.loads(LATEST.read_text())
    assert (
        len(state["ctrl_token_ids"]) == N_CONTROLLED_TOKENS
    ), "checkpoint N_CONTROLLED_TOKENS mismatch"
    ctrl_token_ids = t.tensor(state["ctrl_token_ids"], device=device)
    iter_idx = state["iter"] + 1
    best_score = (
        json.loads(BEST.read_text())["score"] if BEST.exists() else float("-inf")
    )
    print(f"resumed from {LATEST} at iter {iter_idx} (best_score={best_score})")
else:
    iter_idx = 0
    best_score = float("-inf")

# Optimization loop
while True:
    iter_start = time.perf_counter()

    if iter_idx < N_CONTROLLED_TOKENS * 4:
        N_TOPK_REPL = 8  # K in Top-K promising token substitutions
        BATCH_SIZE_OPTIM = 16  # Batch size in optimization
    elif iter_idx < N_CONTROLLED_TOKENS * 8:
        N_TOPK_REPL = 16  # K in Top-K promising token substitutions
        BATCH_SIZE_OPTIM = 32  # Batch size in optimization
    elif iter_idx < N_CONTROLLED_TOKENS * 16:
        N_TOPK_REPL = 32  # K in Top-K promising token substitutions
        BATCH_SIZE_OPTIM = 64  # Batch size in optimization
    else:
        N_TOPK_REPL = 64  # K in Top-K promising token substitutions
        BATCH_SIZE_OPTIM = 128  # Batch size in optimization

    # redundant since model parameters are frozen, but left in as good practice
    model.zero_grad(set_to_none=True)

    # Snapshot the state we're about to score (pre-update) for the checkpoint record.
    ids_scored = ctrl_token_ids
    prompt_str = decode_prompt(ids_scored)

    score_curr, score_grad = compute_score_gradient(ctrl_token_ids)

    # Note: GCG paper https://arxiv.org/abs/2307.15043 formulates the problem as loss minimziation, but we're doing score maximization => no gradient negation
    topk_vals, topk_idxs = score_grad.topk(N_TOPK_REPL, axis=-1)  # (seq, topk)

    with t.inference_mode():
        # For each optimization slot, uniformly pick a random control position and
        # a random replacement from that position's top-k promising substitutions.
        repl_seq_idx = t.randint(
            0, N_CONTROLLED_TOKENS, (BATCH_SIZE_OPTIM,), device=device
        )
        repl_topk_idxidx = t.randint(0, N_TOPK_REPL, (BATCH_SIZE_OPTIM,), device=device)
        # topk_idxidx -> it's an index into topk_idx, so idx-idx

        # (BATCH_SIZE_OPTIM, N_CONTROLLED_TOKENS): each candidate differs from the
        # current prefix in exactly one position.
        candidates = ctrl_token_ids[None].repeat(BATCH_SIZE_OPTIM, 1)
        candidates[t.arange(BATCH_SIZE_OPTIM), repl_seq_idx] = topk_idxs[
            repl_seq_idx, repl_topk_idxidx
        ]

        # Score all candidates in one (chunked) batched forward instead of looping.
        cand_scores = compute_scores_batch(
            trunk,
            model.get_input_embeddings()(candidates),
            sfx_embed,
            sfx_enc["attention_mask"],
            n_sfx_tokens,
            probe_dir,
            LAYER,
            chunk=CAND_CHUNK,
        )
        assert t.isfinite(cand_scores).all(), "Candidate score is not finite (nan?)"
        best_candidate = candidates[t.argmax(cand_scores)]

    iter_time = time.perf_counter() - iter_start

    print(f"{iter_idx=}, current score {score_curr:.4f}, iter_time {iter_time:.2f}s")
    print(f"prompt: {prompt_str}")

    # Checkpoint the scored state and log metrics.
    record = {
        "iter": iter_idx,
        "score": score_curr,
        "ctrl_token_ids": ids_scored.tolist(),
        "prompt": prompt_str,
        "iter_time_s": iter_time,
        "model_id": MODEL_NAME,
        "layer": LAYER,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    save_json_atomic(LATEST, record)
    with open(HISTORY, "a") as f:
        f.write(json.dumps(record) + "\n")
    if score_curr > best_score:
        best_score = score_curr
        save_json_atomic(BEST, record)

    wandb_log(
        {
            "iter": iter_idx,
            "score": score_curr,
            "best_score": best_score,
            "iter_time_s": iter_time,
            "prompt": prompt_str,
        }
    )

    # Update for next loop
    ctrl_token_ids = best_candidate
    iter_idx += 1

    gc.collect()
    t.cuda.empty_cache()
