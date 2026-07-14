"""Equivalence tests for the GCG speedups, on GPT-2 (CPU, offline).

Verifies the utils.py optimizations preserve semantics:
  1. truncate_to_layer leaves hidden_states[LAYER + 1] bit-identical.
  2. compute_scores_batch matches a straightforward per-candidate reference.
  3. Sharding candidates across replicas matches scoring them all at once.
  4. plan_replica_placement groups GPUs correctly for various hardware.

Run: pytest test_speedup_smoke.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import truncate_to_layer, compute_scores_batch, plan_replica_placement

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


# chunk=4 does not divide M=11 evenly, exercising the ragged final chunk.
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
        model.get_input_embeddings()(cands),
        sfx_embed,
        sfx_mask,
        n_sfx,
        probe_dir,
        LAYER,
        chunk=chunk,
    )
    assert t.allclose(got, ref, atol=1e-5)


def test_gradient_flows_to_onehot(model, suffixes, probe_dir):
    """compute_scores_batch is the GCG gradient path: feeding control embeddings
    built from a differentiable one-hot must yield a finite, non-zero gradient on
    that one-hot (mirrors optimize_prompt.compute_score_gradient)."""
    truncate_to_layer(model, LAYER)
    sfx_embed, sfx_mask, n_sfx = suffixes

    g = t.Generator().manual_seed(3)
    ctrl_ids = t.randint(0, model.config.vocab_size, (N_CTRL,), generator=g)

    onehot = t.zeros((N_CTRL, model.config.vocab_size))
    onehot[t.arange(N_CTRL), ctrl_ids] = 1
    onehot.requires_grad = True
    ctrl_embed = (onehot @ model.get_input_embeddings().weight)[None]  # (1, N_CTRL, d)

    scores = compute_scores_batch(
        model.base_model, ctrl_embed, sfx_embed, sfx_mask, n_sfx, probe_dir, LAYER
    )
    assert scores.shape == (1,)
    scores[0].backward()

    grad = onehot.grad
    assert grad is not None
    assert t.isfinite(grad).all()
    assert (grad != 0).any()


@pytest.mark.parametrize("n_shards", [2, 3, 4])
def test_sharded_scoring_matches_single(model, suffixes, probe_dir, n_shards):
    """Splitting candidates across replicas (same model) matches scoring them
    all at once. Mirrors the multi-GPU fan-out without needing extra GPUs;
    n_shards=3 does not divide M=11 evenly. Scores agree up to float batching
    noise, and -- what actually matters -- the argmax winner is unchanged."""
    truncate_to_layer(model, LAYER)
    sfx_embed, sfx_mask, n_sfx = suffixes
    trunk, embed = model.base_model, model.get_input_embeddings()

    M = 11
    g = t.Generator().manual_seed(3)
    cands = t.randint(0, model.config.vocab_size, (M, N_CTRL), generator=g)

    def score(c):
        return compute_scores_batch(
            trunk, embed(c), sfx_embed, sfx_mask, n_sfx, probe_dir, LAYER, chunk=None
        )

    full = score(cands)
    sharded = t.cat([score(s) for s in t.tensor_split(cands, n_shards)])
    assert t.allclose(sharded, full, atol=1e-5)
    assert t.argmax(sharded) == t.argmax(full)


class _FakeParam:
    """Reports a byte footprint without allocating it."""

    def __init__(self, numel, esize=2):
        self._numel, self._esize = numel, esize

    def numel(self):
        return self._numel

    def element_size(self):
        return self._esize


class _StubModel:
    def __init__(self, n_bytes, esize=2):
        self._p = _FakeParam(n_bytes // esize, esize)

    def parameters(self):
        return [self._p]


def _patch_gpus(monkeypatch, count, gib):
    """Fake `count` CUDA devices each with `gib` GiB of memory."""
    monkeypatch.setattr(t.cuda, "device_count", lambda: count)

    def props(i):
        p = type("P", (), {})()
        p.total_memory = int(gib * 1024**3)
        return p

    monkeypatch.setattr(t.cuda, "get_device_properties", props)


def test_plan_replica_placement_autodetect(monkeypatch):
    model = _StubModel(26 * 1024**3)  # ~26 GiB model

    _patch_gpus(monkeypatch, count=8, gib=16)  # needs 2 cards/replica -> 4x2
    assert plan_replica_placement(model) == [[0, 1], [2, 3], [4, 5], [6, 7]]

    _patch_gpus(monkeypatch, count=4, gib=80)  # fits on one big card -> 4x1
    assert plan_replica_placement(model) == [[0], [1], [2], [3]]

    _patch_gpus(monkeypatch, count=2, gib=16)  # only enough for one replica
    assert plan_replica_placement(model) == [[0, 1]]


def test_plan_replica_placement_overrides(monkeypatch):
    model = _StubModel(26 * 1024**3)
    _patch_gpus(monkeypatch, count=8, gib=16)

    # Force fewer replicas than the auto count.
    assert plan_replica_placement(model, n_replicas=2) == [[0, 1], [2, 3]]
    # Force one GPU per replica even though the model would not fit.
    assert plan_replica_placement(model, gpus_per_replica=1) == [[i] for i in range(8)]


def test_plan_replica_placement_no_cuda(monkeypatch):
    monkeypatch.setattr(t.cuda, "device_count", lambda: 0)
    assert plan_replica_placement(_StubModel(1 << 30)) == []
