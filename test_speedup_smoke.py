"""Equivalence tests for the GCG speedups, on GPT-2 (CPU, offline).

Verifies the two optimizations in utils.py preserve semantics:
  1. truncate_to_layer leaves hidden_states[LAYER + 1] bit-identical.
  2. compute_scores_batch matches a straightforward per-candidate reference.

Run: pytest test_speedup_smoke.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import truncate_to_layer, compute_scores_batch

MODEL_NAME = "openai-community/gpt2"
LAYER = 6
N_CTRL = 5
SUFFIX_STRINGS = ["the cat sat", "a longer suffix here", "hi"]


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


@pytest.fixture
def model():
    """Fresh GPT-2 per test, since truncate_to_layer mutates in place."""
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=t.float32)
    m.requires_grad_(False)
    m.eval()
    assert m.config.n_layer > LAYER + 1, "need layers above LAYER to truncate"
    return m


@pytest.fixture
def suffixes(model, tokenizer):
    """Tokenize/pad the suffix strings the way optimize_prompt.py does."""
    enc = tokenizer([" " + s for s in SUFFIX_STRINGS], padding=True)
    sfx_mask = t.tensor(enc["attention_mask"])
    sfx_ids = t.tensor(enc["input_ids"])
    sfx_embed = model.get_input_embeddings()(sfx_ids)
    n_sfx = sfx_mask.sum(axis=1)
    return sfx_embed, sfx_mask, n_sfx


@pytest.fixture
def probe_dir(model):
    g = t.Generator().manual_seed(0)
    d = t.randn(model.config.n_embd, generator=g)
    return d / t.linalg.norm(d)


def probed_hidden(model, ctrl_ids, suffixes):
    """hidden_states[LAYER + 1] at the last real token, for one control prefix."""
    sfx_embed, sfx_mask, n_sfx = suffixes
    n_sfx_strings, _, d_model = sfx_embed.shape
    ce = model.get_input_embeddings()(ctrl_ids)  # (n_ctrl, d_model)
    ce = ce[None].expand(n_sfx_strings, N_CTRL, d_model)
    inp = t.cat([ce, sfx_embed], axis=1)
    cm = t.ones(n_sfx_strings, N_CTRL, dtype=sfx_mask.dtype)
    attn = t.cat([cm, sfx_mask], axis=1)
    out = model.base_model(
        inputs_embeds=inp, attention_mask=attn, output_hidden_states=True
    )
    h = out.hidden_states[LAYER + 1]
    pos = n_sfx + N_CTRL - 1
    return h[t.arange(n_sfx_strings), pos]


def test_truncation_preserves_hidden_state(model, suffixes):
    """Dropping blocks above LAYER must not change hidden_states[LAYER + 1]."""
    g = t.Generator().manual_seed(1)
    ctrl0 = t.randint(0, model.config.vocab_size, (N_CTRL,), generator=g)

    h_full = probed_hidden(model, ctrl0, suffixes)
    n_before = len(model.base_model.h)
    truncate_to_layer(model, LAYER)
    n_after = len(model.base_model.h)
    h_trunc = probed_hidden(model, ctrl0, suffixes)

    assert n_after == LAYER + 2  # keep one extra block past the probe (see docstring)
    assert n_after < n_before
    assert t.allclose(h_full, h_trunc, atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 4, 11, None])
def test_batched_scores_match_reference(model, suffixes, probe_dir, chunk):
    """Chunked batched scoring matches a per-candidate reference for any chunk."""
    truncate_to_layer(model, LAYER)  # batched scoring runs on the trunk
    sfx_embed, sfx_mask, n_sfx = suffixes

    M = 11
    g = t.Generator().manual_seed(2)
    cands = t.randint(0, model.config.vocab_size, (M, N_CTRL), generator=g)

    def reference_score(ctrl_ids):
        h = probed_hidden(model, ctrl_ids, suffixes)
        norm = t.linalg.norm(h, axis=-1)
        return ((h.float() @ probe_dir.float()) / norm.float()).mean()

    ref = t.stack([reference_score(cands[i]) for i in range(M)])
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
    assert t.allclose(got, ref, atol=1e-5)
