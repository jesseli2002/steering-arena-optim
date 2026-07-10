# Score a list of steering prompts against the Season 2 OLMo L24 logistic
# direction, reading activations via nnsight/NDIF from the SAME model the live
# Steering Arena scorer uses. This is the closest possible reproduction of the
# website scores: identical model + layer, the same batched left-padded
# last-token residual read, and the same cosine-steering-shift math.
#
# Why not estimate_probe_score.py? That script reads activations from a LOCAL
# transformers forward, one text at a time (no padding). The live scorer instead
# runs a single BATCHED (left-padded) forward on NDIF. Either of those (local vs
# NDIF numerics, unbatched vs batched padding) can shift the score, which is the
# likeliest reason estimate_probe_score.py doesn't reproduce the leaderboard.
#
# Scoring math (cosine/compose/unit_rows/baseline_unit_rows/shift_and_specificity)
# and the residual read (batch_last_resids) are copied verbatim from the
# read-only steering-arena/app/{scoring,ndif_client}.py -- same convention as
# recover_direction.py / estimate_probe_score.py. steering-arena/ is reference
# only: never imported from, never written to (CLAUDE.md).
#
# The live wiring being mirrored: app/main.py get_scorer() ->
#   base_units = baseline_unit_rows(probes, batch_fn)          # one forward, once
#   shift, z  = shift_and_specificity(seq, probes, batch_fn, base_units, d, eps)
# with batch_fn(texts) = reader.batch_last_resids(texts, layer). Season 2's
# ranked score is the raw `shift`; `specificity_z` is reported alongside (as the
# site does) but is not the leaderboard value.
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

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
    # "your first steering prompt here",
    # "your second steering prompt here",
]


# %%
# ---- Config -----------------------------------------------------------------
# The shipped Season 2 direction (NOT the locally recovered one under data/).
# model_id and layer are read from its metadata, so they always match the d.
DIRECTION_PATH = ARENA_ROOT / "data" / "directions" / "d_olmo3_L24_logistic.npz"
SEASON_FILE = ARENA_ROOT / "data" / "probes" / "season2.json"

# specificity denominator floor -- frozen per season in app/config.py.
SPECIFICITY_EPS = 1e-4

# nnsight remote call retry policy (small on purpose: fail fast for now).
MAX_RETRIES = 4          # total attempts per remote forward (keep in 3..5)
BACKOFF_INITIAL_S = 2.0  # first backoff sleep
BACKOFF_FACTOR = 2.0     # each retry multiplies the sleep by this

REMOTE = True  # read from NDIF (the live path). Set False only for local debug.


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
    d64 = d64 / np.linalg.norm(d64)  # cosine() normalizes d too -- keep exact parity for any d
    mat = unit_rows(batch_resid_fn([compose(seq, p) for p in probes]))
    delta = mat - np.asarray(base_units, dtype=np.float64)          # (P, H)
    # inputs are finite; some BLAS builds emit spurious overflow warnings on @
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shift = float(delta.mean(axis=0) @ d64)
        sigma_null = float(np.linalg.norm(delta)) / float(np.sqrt(delta.size))  # ||Delta||_F/sqrt(P*H)
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
# ---- NDIF residual read, mirroring app/ndif_client.py:batch_last_resids ------
def build_model(model_id: str):
    """nnsight LanguageModel wired to NDIF, same as ResidualReader.build('ndif')."""
    import nnsight
    from nnsight import LanguageModel

    ndif_key = os.getenv("NDIF_API_KEY", "")
    if ndif_key:
        # In-memory only (set_default_api_key would write the key to disk).
        nnsight.CONFIG.API.APIKEY = ndif_key
    elif REMOTE:
        print(
            "[warning] NDIF_API_KEY not set -- relying on nnsight's on-disk config. "
            "Set NDIF_API_KEY (see steering-arena/.env.example) if remote calls 401."
        )
    return LanguageModel(model_id)


def _batch_last_resids_once(model, texts: list[str], layer: int) -> np.ndarray:
    """One batched forward -> (len(texts), hidden) last-token layer-`layer` residuals.

    Left-pads so the last real token sits at index -1 for every row -- the exact
    read the live scorer performs (app/ndif_client.py:batch_last_resids)."""
    import torch

    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    with model.trace(list(texts), remote=REMOTE):
        saved = model.model.layers[layer].output[:, -1, :].save()  # (n, hidden)
    v = saved.value if hasattr(saved, "value") else saved
    return np.asarray(v.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)


def batch_last_resids(model, texts: list[str], layer: int) -> np.ndarray:
    """`_batch_last_resids_once` with exponential backoff on the remote call.

    Retries transient NDIF failures a small number of times, then re-raises so a
    persistent problem surfaces immediately rather than silently looping."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return _batch_last_resids_once(model, texts, layer)
        except Exception as err:  # noqa: BLE001 -- retry any NDIF/transport error for now
            last_err = err
            if attempt == MAX_RETRIES - 1:
                break
            sleep_s = BACKOFF_INITIAL_S * (BACKOFF_FACTOR**attempt)
            print(
                f"[retry] remote forward failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                f"{type(err).__name__}: {err}\n        backing off {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
    raise RuntimeError(
        f"remote forward failed after {MAX_RETRIES} attempts"
    ) from last_err


# %%
# ---- Load direction + probes, build model -----------------------------------
d, meta = load_direction(DIRECTION_PATH)
probes = load_probes(SEASON_FILE)

MODEL_ID = meta["model_id"]
LAYER = int(meta["layer"])

print(f"direction: {DIRECTION_PATH}")
print(f"  model_id={MODEL_ID} layer={LAYER} d_dim={d.shape[0]}")
print(f"probes: {len(probes)} from {SEASON_FILE}")
if meta.get("placeholder"):
    print("[warning] using a PLACEHOLDER direction -- scores are not meaningful.")

model = build_model(MODEL_ID)


def batch_fn(texts: list[str]) -> np.ndarray:
    return batch_last_resids(model, texts, LAYER)


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
base_units = baseline_unit_rows(probes, batch_fn)

results = []
for i, prompt in enumerate(PROMPTS):
    print(f"\n[{i + 1}/{len(PROMPTS)}] scoring: {prompt!r}")
    shift, z = shift_and_specificity(
        prompt, probes, batch_fn, base_units, d, eps=SPECIFICITY_EPS
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
out_path = out_dir / "ndif_prompt_scores.json"
out_path.write_text(
    json.dumps(
        {
            "direction": str(DIRECTION_PATH),
            "direction_meta": meta,
            "season_file": str(SEASON_FILE),
            "model_id": MODEL_ID,
            "layer": LAYER,
            "backend": "ndif" if REMOTE else "local",
            "results": results,
        },
        indent=2,
    )
)
print(f"\nwrote {out_path}")
