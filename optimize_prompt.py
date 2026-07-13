# %%
"""
Optimize the prompt using "Greedy Coordinate Gradients
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import gc

import numpy as np
import torch as t
import einops
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import cos_sim, compose, load_direction, load_prompt_suffixes

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
dtype = t.float32  # for training
if device.type != "cuda":
    print("\033[93mWarning: CUDA not available, using CPU. This will be slow.\033[0m")


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactively estimate the mean Season 2 probe score for "
        "steering prompts, loading the model once and looping over stdin."
    )
    ap.add_argument_group(
        "--smoke",
        action="store_true",
        help="Smoke test on smaller model (GPT-2), run locally to test pipeline",
    )
    return ap.parse_args(argv)


# args = parse_args()
SMOKE = True  # args.smoke

# Load probe direction & season prompt suffix
SEASON_FILE = ARENA_ROOT / "data" / "probes" / "season2.json"
if SMOKE:
    DIRECTION_FILE = REPO_ROOT / "data_local/directions/d_dev_smoke.npz"
else:
    DIRECTION_FILE = ARENA_ROOT / "data/directions/d_olmo3_L24_logistic.npz"

# %%
probe_dir, meta = load_direction(DIRECTION_FILE)
probe_dir = t.tensor(probe_dir / np.linalg.norm(probe_dir), device=device, dtype=dtype)
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
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=dtype, device_map="auto", token=HF_TOKEN
)
if model.device != device:  # in debugging, might use CPU
    model = model.to(device)

model.requires_grad_(False)

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
N_CONTROLLED_TOKENS = 64
N_TOPK_REPL = 256  # K in Top-K promising token substitutions
BATCH_SIZE_OPTIM = 512  # Batch size in optimization

if SMOKE:  # for debugging
    BATCH_SIZE_OPTIM = 32

# %%
# sfx -> suffix tokens
sfx_enc = tokenizer([" " + suffix for suffix in suffixes], padding=True)
n_sfx_tokens = t.tensor(sfx_enc["attention_mask"], device=device).sum(
    axis=1  # (batch, )
)
sfx_tokens = t.tensor(sfx_enc["input_ids"])  # (batch, seq)
BATCH_SIZE = len(n_sfx_tokens)

#  constant
sfx_embed = model.get_input_embeddings()(sfx_tokens.to(device))

# Initialization of optimized value. Random start/restart might do better, but that's for later.
ctrl_token_ids = t.full(
    (N_CONTROLLED_TOKENS,), tokenizer("!")["input_ids"][0], device=device
)

# %%
# Optimization loop


def compute_score(ctrl_token_ids, req_grad: bool):
    """
    Computes mean probe score given tokens in prompt prefix.
    :param ctrl_token_ids: Tokens in prompt prefix
    :req_grad: Set to True to enable tracking gradient of ctrl_tokens_onehot
    """
    # Define controlled tokens
    ctrl_tokens_onehot = t.zeros(  # (seq, d_vocab)
        (N_CONTROLLED_TOKENS, D_VOCAB), dtype=dtype, device=device
    )
    ctrl_tokens_onehot[t.arange(N_CONTROLLED_TOKENS), ctrl_token_ids] = 1

    # in message as a whole, (batch, )
    n_tokens = n_sfx_tokens + N_CONTROLLED_TOKENS

    embed_weights = model.get_input_embeddings().weight  # (d_vocab, d_model)

    ctrl_tokens_onehot.requires_grad = req_grad
    ctrl_embed = ctrl_tokens_onehot @ embed_weights  # (seq, d_model)

    # (batch, seq, d_vocab)
    ctrl_embed_expand = ctrl_embed[None].expand(BATCH_SIZE, *ctrl_embed.shape)
    tokens_embed = t.cat([ctrl_embed_expand, sfx_embed], axis=1)

    # Build attention mask
    sfx_mask = t.tensor(sfx_enc["attention_mask"], device=device)  # (batch, sfx_seq)
    ctrl_mask = t.ones(
        BATCH_SIZE, N_CONTROLLED_TOKENS, device=device, dtype=sfx_mask.dtype
    )
    attn_mask = t.cat([ctrl_mask, sfx_mask], axis=1)  # (batch, seq)
    result = model(
        inputs_embeds=tokens_embed, output_hidden_states=True, attention_mask=attn_mask
    )
    probed_layer = result.hidden_states[LAYER + 1]  # (batch, seq, d_model)
    probed_acts = probed_layer[t.arange(BATCH_SIZE), n_tokens - 1]  # (batch, d_model)
    # TODO: Check with steering-arena author their layer numbering convention for probe.

    norm = t.linalg.norm(probed_acts, axis=-1)  # (batch,)
    scores = (
        einops.einsum(probed_acts, probe_dir, "batch d_model, d_model -> batch") / norm
    )
    return ctrl_tokens_onehot, scores.mean()


# TODO: Might be faster to batch over BATCH_SIZE_OPTIM in addition to BATCH_SIZE
# TODO: Functionality to load from checkpoint
iter_idx = 0

while True:
    # redundant since model parameters are frozen, but left in as good practice
    model.zero_grad(set_to_none=True)

    ctrl_tokens_onehot, mean_score_curr = compute_score(ctrl_token_ids, req_grad=True)

    print(f"{iter_idx=}, current score {mean_score_curr}")
    print(f"prompt: ")  # TODO: decode ctrl_token_ids to prompt string

    mean_score_curr.backward()  # Maximize score => pick top k gradients

    # Note: GCG paper https://arxiv.org/abs/2307.15043 formulates the problem as loss minimziation, but we're doing score maximization => no gradient negation
    topk_vals, topk_idxs = ctrl_tokens_onehot.grad.topk(  # (seq, topk)
        N_TOPK_REPL, axis=-1
    )

    with t.inference_mode():
        # For each optimization batch, uniformly select a random seq index & random value from topk
        repl_seq_idx = t.randint(0, topk_idxs.shape[0], (BATCH_SIZE_OPTIM,))
        repl_topk_idxidx = t.randint(0, topk_idxs.shape[1], (BATCH_SIZE_OPTIM,))
        # topk_idxidx -> it's an index into topk_idx, so idx-idx

        candidates = []
        cand_scores = []
        for seq_idx, topk_idxidx in tqdm(zip(repl_seq_idx, repl_topk_idxidx)):
            candidate = ctrl_token_ids.clone()  # (seq, )
            candidate[seq_idx] = topk_idxs[seq_idx, topk_idxidx]
            _, cand_score = compute_score(candidate, req_grad=False)

            assert t.isfinite(cand_score), "Candidate score is not finite (nan?)"
            candidates.append(candidate)
            cand_scores.append(cand_score)

        cand_scores = t.tensor(cand_scores)
        best_candidate = candidates[t.argmax(cand_scores)]

    # Update for next loop
    ctrl_token_ids = best_candidate
    iter_idx += 1


gc.collect()
t.cuda.empty_cache()
