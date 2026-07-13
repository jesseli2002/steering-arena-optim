"""Smoke/equivalence test for the GCG speedups, on GPT-2 (CPU, offline).

Checks that the two optimizations preserve semantics:
  1. truncate_to_layer leaves hidden_states[LAYER + 1] bit-identical.
  2. compute_scores_batch matches a straightforward per-candidate reference.

Run: HF_HUB_OFFLINE=1 python test_speedup_smoke.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import truncate_to_layer, compute_scores_batch

MODEL_NAME = "openai-community/gpt2"
LAYER = 6
N_CTRL = 5

t.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=t.float32)
model.requires_grad_(False)
model.eval()

D_VOCAB = model.config.vocab_size
D_MODEL = model.config.n_embd
N_LAYERS = model.config.n_layer
assert N_LAYERS > LAYER + 1, "test needs layers above LAYER to actually truncate"


def build_suffixes(strings):
    """Tokenize/pad a few suffix strings the way optimize_prompt.py does."""
    enc = tok([" " + s for s in strings], padding=True)
    mask = t.tensor(enc["attention_mask"])
    ids = t.tensor(enc["input_ids"])
    sfx_embed = model.get_input_embeddings()(ids)
    n_sfx = mask.sum(axis=1)
    return sfx_embed, mask, n_sfx


suffix_strings = ["the cat sat", "a longer suffix here", "hi"]
sfx_embed, sfx_mask, n_sfx = build_suffixes(suffix_strings)
S = len(suffix_strings)

probe_dir = t.randn(D_MODEL)
probe_dir = probe_dir / t.linalg.norm(probe_dir)


# --- Test 1: truncation preserves the probed hidden state ---------------------
def probed_hidden(m, ctrl_ids):
    """hidden_states[LAYER + 1] at the last real token, for one prefix."""
    ce = m.get_input_embeddings()(ctrl_ids)  # (n_ctrl, d_model)
    ce = ce[None].expand(S, N_CTRL, D_MODEL)
    inp = t.cat([ce, sfx_embed], axis=1)
    cm = t.ones(S, N_CTRL, dtype=sfx_mask.dtype)
    attn = t.cat([cm, sfx_mask], axis=1)
    out = m.base_model(
        inputs_embeds=inp, attention_mask=attn, output_hidden_states=True
    )
    h = out.hidden_states[LAYER + 1]
    pos = n_sfx + N_CTRL - 1
    return h[t.arange(S), pos]


ctrl0 = t.randint(0, D_VOCAB, (N_CTRL,))
h_full = probed_hidden(model, ctrl0)
n_before = len(model.base_model.h)
truncate_to_layer(model, LAYER)
n_after = len(model.base_model.h)
h_trunc = probed_hidden(model, ctrl0)

assert n_before == N_LAYERS and n_after == LAYER + 2, (n_before, n_after)
max_err = (h_full - h_trunc).abs().max().item()
print(f"[truncation] blocks {n_before} -> {n_after}, hidden max_err={max_err:.2e}")
assert t.allclose(
    h_full, h_trunc, atol=1e-5
), "truncation changed the probed hidden state"


# --- Test 2: batched scoring matches a per-candidate reference -----------------
def reference_score(ctrl_ids):
    """Original-style single-candidate score, using the (truncated) trunk."""
    h = probed_hidden(model, ctrl_ids)  # (S, d_model)
    norm = t.linalg.norm(h, axis=-1)
    return ((h.float() @ probe_dir.float()) / norm.float()).mean()


M = 11
cands = t.randint(0, D_VOCAB, (M, N_CTRL))
ref = t.stack([reference_score(cands[i]) for i in range(M)])

for chunk in (1, 4, M):
    got = compute_scores_batch(
        model.base_model,
        model.get_input_embeddings(),
        cands,
        sfx_embed,
        sfx_mask,
        n_sfx,
        probe_dir,
        LAYER,
        chunk=chunk,
    )
    err = (got - ref).abs().max().item()
    print(f"[batched] chunk={chunk:>2} vs reference max_err={err:.2e}")
    assert t.allclose(got, ref, atol=1e-5), f"batched scores differ (chunk={chunk})"

print("\nALL CHECKS PASSED")
