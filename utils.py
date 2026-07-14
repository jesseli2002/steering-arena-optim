import json
from pathlib import Path

import numpy as np
import torch as t


# %%
def cos_sim(a: np.ndarray, b: np.ndarray):
    """
    Compute cosine similarity of two vectors.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine of a zero vector is undefined")
    return np.dot(a, b) / (na * nb)


def compose(seq: str, probe: str) -> str:
    """How a candidate sequence is prepended to a probe. Fixed for determinism."""
    return f"{seq} {probe}"


def load_direction(path) -> tuple[np.ndarray, dict]:
    """Load `d` (float32) and its metadata from a d_<version>.npz file."""
    data = np.load(path, allow_pickle=True)
    d = np.asarray(data["d"], dtype=np.float32)
    meta = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"]))
    return d, meta


def load_prompt_suffixes(path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["prompts"] if isinstance(obj, dict) else list(obj)


def truncate_to_layer(model, layer: int):
    """
    Monkey-patches model in-place, dropping transformer blocks above `layer` so the forward stops right after
    producing hidden_states[layer + 1].

    The probe only reads one layer's residual stream, so running the remaining
    blocks (and the final norm / lm_head, which are skipped by calling
    `model.base_model` for the forward) is wasted work. Mutates and returns
    `model`. Works for Llama/OLMo-style trunks (`.layers`) and GPT-2 (`.h`).

    We keep `layer + 2` blocks, not `layer + 1`: the model applies the final
    norm to the *last* entry of `hidden_states`, so we keep one extra raw block
    output so that `hidden_states[layer + 1]` stays the pre-norm block output the
    probe reads (identical to the untruncated model).
    """
    trunk = model.base_model
    if hasattr(trunk, "layers"):
        attr = "layers"
    elif hasattr(trunk, "h"):
        attr = "h"
    else:
        raise ValueError(f"cannot locate decoder blocks on {type(trunk).__name__}")
    blocks = getattr(trunk, attr)
    assert layer + 1 <= len(blocks), f"layer {layer} exceeds model depth {len(blocks)}"
    keep = min(layer + 2, len(blocks))
    setattr(trunk, attr, blocks[:keep])
    for name in ("num_hidden_layers", "n_layer"):
        if hasattr(model.config, name):
            setattr(model.config, name, keep)
    return model


def compute_scores_batch(
    trunk,
    ctrl_embed,
    sfx_embed,
    sfx_mask,
    n_sfx_tokens,
    probe_dir,
    layer: int,
    chunk=None,
):
    """Mean-over-suffix probe score for each of n_cand candidate prefixes.

    Each candidate prefix is scored against all n_sfx suffixes in one (chunked)
    batched forward -- packing candidates into a single forward instead of
    looping is where the speedup comes from. All candidates share the same fixed
    suffixes; only the control-token prefix varies.

    Takes control embeddings directly (rather than token ids + an embedding
    layer) so the two callers differ only in how they produce ctrl_embed: the
    batch candidate scorer feeds an integer embed() lookup, while the
    single-prefix GCG gradient path (compute_score_gradient) feeds a differentiable
    one-hot @ embedding-weight product. This function is differentiable w.r.t.
    ctrl_embed, so gradients flow back through it in that case.

    :param trunk: model.base_model (returns hidden_states, no lm_head)
    :param ctrl_embed: (n_cand, ctrl_seq, d_model) candidate prefix embeddings
    :param sfx_embed: (n_sfx, sfx_seq, d_model) precomputed suffix embeddings
    :param sfx_mask: (n_sfx, sfx_seq) suffix attention mask
    :param n_sfx_tokens: (n_sfx,) real (unpadded) suffix token counts
    :param probe_dir: (d_model,) unit probe direction
    :param layer: probe layer index (reads hidden_states[layer + 1])
    :param chunk: candidates per forward pass (memory knob); None = all at once
    :returns: (n_cand,) float32 mean-over-suffix score per candidate
    """
    device = sfx_embed.device
    # n_cand -> # of candidates; ctrl_seq -> # of controlled tokens
    # n_sfx -> # of suffixs to test; sfx_seq -> sequence index in suffix
    n_cand, ctrl_seq, _ = ctrl_embed.shape
    n_sfx, sfx_seq, d_model = sfx_embed.shape
    seq = ctrl_seq + sfx_seq

    # Right padding => last real token sits at ctrl_seq + n_sfx - 1.
    gather_pos = n_sfx_tokens + ctrl_seq - 1  # (n_sfx,)
    ctrl_mask = t.ones(n_sfx, ctrl_seq, dtype=sfx_mask.dtype)

    if chunk is None:
        chunk = n_cand

    score_chunks = []
    for ce in t.split(ctrl_embed, chunk):
        c = ce.shape[0]  # chunk size; may be smaller than `chunk` for last iter

        ctrl_e = ce[:, None].expand(c, n_sfx, ctrl_seq, d_model)
        sfx_e = sfx_embed[None].expand(c, n_sfx, sfx_seq, d_model)
        inp = t.cat([ctrl_e, sfx_e], dim=2).reshape(c * n_sfx, seq, d_model)

        # Build attention mask
        cm = ctrl_mask[None].expand(c, n_sfx, ctrl_seq)
        sm = sfx_mask[None].expand(c, n_sfx, sfx_seq)
        attn = t.cat([cm, sm], dim=2).reshape(c * n_sfx, seq)

        out = trunk(inputs_embeds=inp, attention_mask=attn, output_hidden_states=True)
        h = out.hidden_states[layer + 1]  # (c*n_sfx, seq, d_model)

        # recall probe_dir is on CPU and float32
        pos = gather_pos[None].expand(c, n_sfx).reshape(c * n_sfx)
        acts = h[t.arange(c * n_sfx), pos].cpu().float()  # (c*n_sfx, d_model)
        norm = t.linalg.norm(acts, dim=-1)

        sc = (acts @ probe_dir) / norm  # (c*n_sfx,)
        score_chunks.append(sc.reshape(c, n_sfx).mean(dim=1))

    return t.cat(score_chunks)
