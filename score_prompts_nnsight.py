# Score a list of steering prompts against the Season 2 OLMo L24 logistic
# direction, reading activations through nnsight -- the same wrapper the live
# Steering Arena scorer uses -- but with nnsight's LOCAL backend (remote=False)
# instead of NDIF.
#
# The point is minimal divergence from the live scorer: the activation-reading
# code is the exact same nnsight trace (model.model.layers[L].output[:, -1, :],
# batched + left-padded) copied verbatim from app/ndif_client.py:batch_last_resids,
# and the cosine-steering-shift math is copied verbatim from app/scoring.py. The
# ONLY difference from live is where the forward runs: locally on this box's GPU
# rather than on NDIF (needed because the NDIF key can't reach this model). Same
# nnsight, same trace, same HF model underneath -- so this is the closest local
# reproduction of the leaderboard, and closer than estimate_probe_score.py
# (which reimplements the read via transformers output_hidden_states, unbatched
# and unpadded).
#
# steering-arena/ is reference only: never imported from, never written to
# (CLAUDE.md) -- hence the verbatim copies below.
import json
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from nnsight import LanguageModel

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
ARENA_ROOT = REPO_ROOT / "steering-arena"
if not ARENA_ROOT.is_dir() and os.environ.get("STEERING_ARENA_DIR"):
    ARENA_ROOT = Path(os.environ["STEERING_ARENA_DIR"]).resolve()
assert (
    ARENA_ROOT.is_dir()
), f"expected read-only steering-arena clone at {ARENA_ROOT} (or set STEERING_ARENA_DIR)"


# %%
# ---- Prompts to score -------------------------------------------------------
# Fill these in. Each entry is scored independently as the steering `seq` that
# gets prepended to every Season 2 probe via compose(seq, p) = f"{seq} {p}".
PROMPTS: list[str] = [
    "You are two months old.",
    "You are two years old.",
    "You are three years old.",
    "You will respond in a short sentence with kindnesz respect compassion and my love.",
    "You will respond in a short sentence with kindnesz respect and love.",
]


# %%
# ---- Config -----------------------------------------------------------------
# The shipped Season 2 direction (NOT the locally recovered one under data/).
# model_id and layer are read from its metadata, so they always match the d.
DIRECTION_PATH = ARENA_ROOT / "data" / "directions" / "d_olmo3_L24_logistic.npz"
SEASON_FILE = ARENA_ROOT / "data" / "probes" / "season2.json"

# specificity denominator floor -- frozen per season in app/config.py.
SPECIFICITY_EPS = 1e-4

# Run the nnsight forward locally on this box. NDIF (remote=True) is unavailable
# for this model, which is the whole reason this script exists; flipping this to
# True would additionally need an NDIF key with access to the model.
REMOTE = False


# %%
# ---- Scoring math, copied verbatim from steering-arena/app/scoring.py --------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine of a zero vector is undefined")
    return float(np.dot(a, b) / (na * nb))


def compose(seq: str, probe: str) -> str:
    """How a candidate sequence is prepended to a probe. Fixed for determinism."""
    return f"{seq} {probe}"


def unit_rows(mat: np.ndarray) -> np.ndarray:
    """Rows normalized to unit length, float64. Raises on a zero row (matches cosine())."""
    m = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("cannot unit-normalize a zero activation row")
    return m / norms


def baseline_unit_rows(probes, batch_resid_fn) -> np.ndarray:
    """(P, H) unit-normalized baseline activations -- ONE batched forward, precompute
    per (season, probes)."""
    return unit_rows(batch_resid_fn(list(probes)))


def shift_and_specificity(
    seq: str,
    probes,
    batch_resid_fn,
    base_units: np.ndarray,
    d: np.ndarray,
    *,
    eps: float = SPECIFICITY_EPS,
) -> tuple[float, float]:
    """(raw steering shift, closed-form specificity z) from one batched forward.

    `shift` is numerically identical to the per-probe cosine-difference mean:
    mean_i(cos(m_i,d) - cos(b_i,d)) = mean_i((m_i_hat - b_i_hat) . d) = delta_bar . d
    for unit d.
    """
    d64 = np.asarray(d, dtype=np.float64)
    d64 = d64 / np.linalg.norm(
        d64
    )  # cosine() normalizes d too -- keep exact parity for any d
    mat = unit_rows(batch_resid_fn([compose(seq, p) for p in probes]))
    delta = mat - np.asarray(base_units, dtype=np.float64)  # (P, H)
    # inputs are finite; some BLAS builds emit spurious overflow warnings on @
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shift = float(delta.mean(axis=0) @ d64)
        sigma_null = float(np.linalg.norm(delta)) / float(
            np.sqrt(delta.size)
        )  # ||Delta||_F/sqrt(P*H)
    z = shift / max(sigma_null, eps)
    return shift, float(z)


def load_direction(path) -> tuple[np.ndarray, dict]:
    """Load `d` (float32) and its metadata from a d_<version>.npz file."""
    data = np.load(path, allow_pickle=True)
    d = np.asarray(data["d"], dtype=np.float32)
    meta = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"]))
    return d, meta


def load_probes(path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["prompts"] if isinstance(obj, dict) else list(obj)


# %%
# ---- Load direction + probes ------------------------------------------------
d, meta = load_direction(DIRECTION_PATH)
probes = load_probes(SEASON_FILE)

MODEL_ID = meta["model_id"]
LAYER = int(meta["layer"])

print(f"direction: {DIRECTION_PATH}")
print(f"  model_id={MODEL_ID} layer={LAYER} d_dim={d.shape[0]}")
print(f"probes: {len(probes)} from {SEASON_FILE}")
if meta.get("placeholder"):
    print("[warning] using a PLACEHOLDER direction -- scores are not meaningful.")


# %%
# ---- Local nnsight model ----------------------------------------------------
# device_map/dtype/token flow through nnsight into AutoModelForCausalLM.from_pretrained
# (the same route the arena's local dev path uses via LanguageModel(model_id,
# device_map=...)). Weights dispatch lazily on the first trace below.
has_cuda = torch.cuda.is_available()
if not has_cuda:
    print(
        "\033[93mWarning: CUDA not available, running on CPU. This will be slow.\033[0m"
    )
dtype = torch.bfloat16 if has_cuda else torch.float32

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN"

model = LanguageModel(MODEL_ID, device_map="auto", dtype=dtype, token=HF_TOKEN)

# config is loaded eagerly (meta init), so these are available pre-dispatch.
H = int(model.config.hidden_size)
NUM_LAYERS = int(model.config.num_hidden_layers)
assert H == d.shape[0], f"direction dim {d.shape[0]} != model hidden size {H}"
assert (
    0 <= LAYER < NUM_LAYERS
), f"layer {LAYER} out of range for {NUM_LAYERS}-layer model"


def batch_last_resids(texts: list[str]) -> np.ndarray:
    """Last-token layer-`LAYER` residual for a BATCH of texts in ONE forward pass
    -> (n, H). Copied verbatim from app/ndif_client.py:batch_last_resids (layer
    pinned to LAYER, remote=REMOTE).

    Left-pads so the last real token sits at index -1 for every row (HF derives
    position_ids/attention from the mask, so this matches the unbatched value).
    One `.save()` of the batched layer output."""
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    with model.trace(list(texts), remote=REMOTE):
        saved = model.model.layers[LAYER].output[:, -1, :].save()  # (n, H)
    v = saved.value if hasattr(saved, "value") else saved
    return np.asarray(v.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)


# %%
# ---- Score every prompt -----------------------------------------------------
# Baselines cos(R_L(p), d) depend only on (probes, d): one batched forward, once,
# reused for every prompt -- exactly as app/main.py precomputes base_units.
if not PROMPTS:
    raise SystemExit(
        "PROMPTS is empty -- add steering prompts to the PROMPTS list at the top "
        "of this file, then re-run."
    )

print("\ncomputing probe baselines (one batched forward)...")
base_units = baseline_unit_rows(probes, batch_last_resids)

results = []
for i, prompt in enumerate(PROMPTS):
    print(f"\n[{i + 1}/{len(PROMPTS)}] scoring: {prompt!r}")
    shift, z = shift_and_specificity(
        prompt, probes, batch_last_resids, base_units, d, eps=SPECIFICITY_EPS
    )
    print(f"  probe score (cosine_steering_shift) = {shift:+.6f}")
    print(f"  specificity_z                       = {z:+.2f}")
    results.append({"prompt": prompt, "shift": shift, "specificity_z": z})

# %%
# ---- Summary + save ---------------------------------------------------------
print("\n=== summary (sorted by score, high to low) ===")
for r in sorted(results, key=lambda r: r["shift"], reverse=True):
    print(f"  {r['shift']:+.6f}  (z={r['specificity_z']:+.2f})  {r['prompt']!r}")

out_dir = REPO_ROOT / "data" / "scores"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "local_prompt_scores.json"
out_path.write_text(
    json.dumps(
        {
            "direction": str(DIRECTION_PATH),
            "direction_meta": meta,
            "season_file": str(SEASON_FILE),
            "model_id": MODEL_ID,
            "layer": LAYER,
            "backend": "nnsight_local",
            "results": results,
        },
        indent=2,
    )
)
print(f"\nwrote {out_path}")
