import copy
import json
import math
from pathlib import Path

import numpy as np
import torch as t
from jaxtyping import Float, Int


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
    ctrl_embed: Float[t.Tensor, "n_cand ctrl_seq d_model"],
    sfx_embed: Float[t.Tensor, "n_sfx sfx_seq d_model"],
    sfx_mask: Int[t.Tensor, "n_sfx sfx_seq"],
    n_sfx_tokens: Int[t.Tensor, "n_sfx"],
    probe_dir: Float[t.Tensor, "d_model"],
    layer: int,
    chunk: int | None = None,
) -> Float[t.Tensor, "n_cand"]:
    """Mean-over-suffix probe score for each of n_cand candidate prefixes.

    Each candidate prefix is scored against all n_sfx suffixes in one (chunked) batched forward. All candidates share the same fixed
    suffixes; only the control-token prefix varies.

    This function is differentiable w.r.t. ctrl_embed, so gradients can flow back through it.

    :param trunk: model.base_model (returns hidden_states, no lm_head)
    :param ctrl_embed: candidate prefix embeddings; dtype/device match the model's
        embedding layer (e.g. bfloat16 on the model's device). May require grad --
        this is the tensor gradients flow back through.
    :param sfx_embed: precomputed suffix embeddings; same dtype/device as ctrl_embed.
    :param sfx_mask: suffix attention mask; integer dtype, CPU (as produced by the
        tokenizer -- not moved to the model's device).
    :param n_sfx_tokens: real (unpadded) suffix token counts; integer dtype, CPU.
    :param probe_dir: unit probe direction; float32, CPU (probe scoring is done on
        CPU by convention).
    :param layer: probe layer index (reads hidden_states[layer + 1])
    :param chunk: candidates per forward pass (memory knob); None = all at once
    :returns: mean-over-suffix score per candidate; float32, CPU (activations are
        moved to CPU before the dot product with probe_dir).
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


def plan_replica_placement(
    model, n_replicas=None, gpus_per_replica=None, mem_fraction=0.9
):
    """Decide GPU groups for data-parallel replicas from runtime hardware.

    Detects device count and per-card capacity so the same code adapts to
    2/4/8 GPUs of any size; all knobs are overridable.

    :param model: the (already truncated) model, used to measure its footprint
    :param n_replicas: force replica count; None => n_gpus // gpus_per_replica
    :param gpus_per_replica: force GPUs per replica; None => from capacity
    :param mem_fraction: usable fraction of each card (headroom for activations)
    :returns: list of GPU-index groups, one per replica, e.g. [[0, 1], [2, 3]].
        Empty list => no CUDA => caller should use the single-model path.
    """
    n_gpus = t.cuda.device_count()
    if n_gpus == 0:
        return []
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # min() is conservative if the cards are heterogeneous.
    budget = (
        min(t.cuda.get_device_properties(i).total_memory for i in range(n_gpus))
        * mem_fraction
    )
    if gpus_per_replica is None:
        gpus_per_replica = max(1, math.ceil(model_bytes / budget))
    if gpus_per_replica > n_gpus:
        raise RuntimeError(
            f"model needs ~{gpus_per_replica} GPUs but only {n_gpus} present"
        )
    if n_replicas is None:
        n_replicas = n_gpus // gpus_per_replica
    return [
        list(range(r * gpus_per_replica, (r + 1) * gpus_per_replica))
        for r in range(n_replicas)
    ]


def build_replicas(model, groups, mem_fraction=0.9):
    """Clone a (truncated) model onto each GPU group for data parallelism.

    Loads/truncates happen once by the caller; this pays only the clone cost.
    Each replica is dispatched (pipelined) across the GPUs of its group.

    :param model: the truncated model to clone (moved to CPU as the source)
    :param groups: GPU-index groups from plan_replica_placement
    :param mem_fraction: usable fraction of each card for the device map
    :returns: list of dispatched replica models, one per group
    """
    from accelerate import dispatch_model, infer_auto_device_map

    model = model.to("cpu")  # truncated weights now in host RAM
    t.cuda.empty_cache()
    replicas = []
    for grp in groups:
        rep = copy.deepcopy(model)
        max_mem = {
            g: int(t.cuda.get_device_properties(g).total_memory * mem_fraction)
            for g in grp
        }
        dev_map = infer_auto_device_map(
            rep,
            max_memory=max_mem,
            no_split_module_classes=model._no_split_modules,
        )
        replicas.append(dispatch_model(rep, device_map=dev_map))
    del model
    return replicas
