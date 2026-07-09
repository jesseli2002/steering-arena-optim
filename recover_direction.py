# %%
# Recover the Season 2 probe direction (`d_olmo3_L24_logistic.npz`) from scratch,
# locally, against the public extraction recipe in `steering-arena/scripts/`
# (extract_direction.py + layer_sweep.py), then validate it with the full
# validate_direction.py + confound_audit.py suite reimplemented against locally
# computed activations (no NDIF). See PROJECT_SPEC.md §11.3-11.4 in steering-arena/
# for the spec this recipe follows.
#
# `steering-arena/` is read-only reference (CLAUDE.md) — never written to. All
# outputs land under `steering-arena-optim/data/`.
import datetime as dt
import hashlib
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t
from dotenv import load_dotenv
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# %%
# ── Wire up read-only imports from steering-arena/scripts/ ──────────────────
# The reference scripts pull in `app.config.settings` (needs pydantic-settings,
# not installed in this dev sandbox) and `app.ndif_client.ResidualReader` (NDIF
# machinery) purely as unused module-level imports — we never call NDIF or read
# `settings` here (activations come from a local transformers forward pass, not
# ResidualReader). Stub both so we can import the *pure* pieces we do reuse
# (LONG/SHORT/POS/NEG, APPROACH/AVOID/CONTROL_PAIRS, estimate_direction, unit,
# compose) verbatim instead of re-typing them and risking drift.
REPO_ROOT = Path(__file__).resolve().parent
ARENA_ROOT = REPO_ROOT / "steering-arena"

if not ARENA_ROOT.is_dir() and os.environ.get("STEERING_ARENA_DIR"):
    # Convention is a sibling `steering-arena/` checkout; this override exists purely
    # for isolated dev sandboxes (e.g. a git worktree) where that sibling path isn't
    # checked out (it's gitignored) — never needed on the real run box.
    ARENA_ROOT = Path(os.environ["STEERING_ARENA_DIR"]).resolve()
assert (
    ARENA_ROOT.is_dir()
), f"expected read-only steering-arena clone at {ARENA_ROOT} (or set STEERING_ARENA_DIR)"


from recover_direction_prompts import (
    LONG,
    SHORT,
    POS,
    NEG,
    APPROACH,
    AVOID,
    POS_AUDIT,
    NEG_AUDIT,
    CONTROL_PAIRS,
)


# Copy implementations from  extract_direction.py
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def estimate_direction(method, pos_chosen, pos_rejected):
    """Direction (raw, un-normalized) separating chosen from rejected at one layer.

    meandiff — mean of per-pair differences (mass-mean); logistic/lda — sklearn probe
    coefficient (covariance-aware). LDA/logistic are offline-only (sklearn not shipped
    to the Space). All return a vector oriented chosen>rejected.

    pos_chosen and pos_rejected are shape (n_data, d_model)
    """
    if method == "meandiff":
        return (pos_chosen - pos_rejected).mean(axis=0)
    X = np.vstack([pos_chosen, pos_rejected])
    y = np.r_[np.ones(len(pos_chosen)), np.zeros(len(pos_rejected))]
    with np.errstate(
        over="ignore", invalid="ignore", divide="ignore"
    ):  # spurious BLAS float warnings
        if method == "lda":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        elif method == "logistic":  # logistic
            clf = LogisticRegression(C=0.1, max_iter=4000).fit(X, y)
        else:
            raise AssertionError(f"Method {method} not recognized")

    return np.asarray(clf.coef_[0], dtype=np.float64)


def compose(prompt, completion):
    return f"{prompt} {completion}"


# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if device != t.device("cuda"):
    print("\033[93mWarning: CUDA not available, using CPU. This will be slow.\033[0m")
dtype = t.bfloat16

if t.cuda.is_available():
    t.cuda.set_device(0)  # ??? Possibly needed?

# %%
# Set to true when developing locally (to test pipeline); smaller models are used to fit on the GPU
LOCAL_DEV = True

if LOCAL_DEV:
    MODEL_NAME = "openai-community/gpt2-xl"  # 1.5B params, ~6GB
    LAYER = 6  # arbitrary for the smoke test
    max_memory = None
else:
    MODEL_NAME = "allenai/Olmo-3-1125-32B"
    LAYER = 24  # pinned by steering-arena/app/config.py (Season 2)
    max_memory = None
    # max_memory = { # TODO - fill out once VastAI setup is chosen
    #     0: "5GiB",  # reserve room for experiments
    #     1: "10GiB",
    #     2: "10GiB",
    #     3: "9GiB",
    # }

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
    max_memory=max_memory,
    token=HF_TOKEN,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %%
demo_inputs = tokenizer("hello world!", return_tensors="pt").to(model.device)
demo_generated = model.generate(**demo_inputs, max_new_tokens=3)
print(tokenizer.decode(demo_generated[0], skip_special_tokens=True))

# %%
NUM_LAYERS = model.config.n_layer
D_MODEL = model.config.n_embd

print(f"Model: {MODEL_NAME}")
print(f"Layers: {NUM_LAYERS}, Hidden dim: {D_MODEL}")
assert (
    0 <= LAYER < NUM_LAYERS
), f"layer {LAYER} out of range for {NUM_LAYERS}-layer model"

# %%
# ── Output locations (all under steering-arena-optim/, never steering-arena/) ──
CACHE_DIR = REPO_ROOT / "data" / "cache" / "acts" / MODEL_NAME.replace("/", "_")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO_ROOT / "data" / "directions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
D_VERSION = "olmo3_L24_logistic"
OUT_NPZ = OUT_DIR / "d_olmo3_L24_logistic.recovered.npz"
OUT_REPORT = OUT_DIR / "d_olmo3_L24_logistic.recovered.report.json"
SHIPPED_NPZ = ARENA_ROOT / "data" / "directions" / "d_olmo3_L24_logistic.npz"
SHIPPED_AUDIT = (
    ARENA_ROOT / "data" / "directions" / "d_olmo3_L24_logistic.confound_audit.json"
)
SEED_PAIRS = ARENA_ROOT / "data" / "seed_pairs.jsonl"

if LOCAL_DEV:
    CACHE_DIR = (
        REPO_ROOT
        / "data"
        / "cache"
        / "acts"
        / (MODEL_NAME.replace("/", "_") + "_LOCAL_DEV")
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NPZ = OUT_DIR / "d_dev_smoke.npz"
    OUT_REPORT = OUT_DIR / "d_dev_smoke.report.json"


# %%
# ── Activation extraction: local transformers forward, disk-cached per (model,
# layer, text), one text at a time (no padding — matches nnsight's plain
# .trace(text), avoids left/right-pad edge cases). ──────────────────────────
def _cache_fp(text: str, layer: int) -> Path:
    key = hashlib.sha256(
        f"{MODEL_NAME}\x00L{layer}\x00{text}".encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / f"{key}.npy"


@t.no_grad()
def last_token_resid(text: str, layer: int) -> np.ndarray:
    """Layer-`layer` residual stream at the last token of `text`, float32 numpy.

    hidden_states[0] is the embedding output; hidden_states[i+1] is decoder layer
    i's output (HF convention) — this is what nnsight's model.model.layers[i].output
    refers to, verified against nnsight's local backend on gpt2 in the smoke test.
    """
    fp = _cache_fp(text, layer)  # cache file path
    if fp.exists():
        return np.load(fp)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    vec = out.hidden_states[layer + 1][0, -1, :].to(t.float32).cpu().numpy()
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, vec)
    tmp.replace(fp)
    return vec


def read_many(texts: list[str], layer: int, label: str) -> np.ndarray:
    """
    Get activations after layer `layer` for each text in `texts.
    :param label: Used for printing only
    """
    out = np.empty((len(texts), D_MODEL), dtype=np.float64)  # returned value
    n_hit = 0
    for i, text in enumerate(tqdm(texts, desc=label)):
        hit = _cache_fp(text, layer).exists()
        n_hit += hit
        out[i] = last_token_resid(text, layer)
    print(f"  [{label}] {len(texts)} texts ({n_hit} cache hits)")
    return out


# %%
# ── Reproduce the exact extraction recipe ──
rows = [
    json.loads(l)
    for l in SEED_PAIRS.read_text(encoding="utf-8").splitlines()
    if l.strip()
]
print(f"loaded {len(rows)} seed pairs from {SEED_PAIRS}")

chosen_texts = [compose(r["prompt"], r["chosen"]) for r in rows]
rejected_texts = [compose(r["prompt"], r["rejected"]) for r in rows]

print(f"reading chosen/rejected activations at layer {LAYER}…")
chosen = read_many(chosen_texts, LAYER, "chosen")
rejected = read_many(rejected_texts, LAYER, "rejected")
assert (
    np.isfinite(chosen).all() and np.isfinite(rejected).all()
), "non-finite activations — aborting"

N = len(rows)
SPLIT_SEED = 0  # guessed default
VAL_FRAC = 0.2
rng = np.random.default_rng(SPLIT_SEED)
idx = rng.permutation(N)
n_val = max(1, int(round(N * VAL_FRAC)))
val_idx, train_idx = idx[:n_val], idx[n_val:]
print(f"split: {len(train_idx)} train / {len(val_idx)} val (seed={SPLIT_SEED})")

d_raw = estimate_direction("logistic", chosen[train_idx], rejected[train_idx])

# Orthogonalization (probe should be orthogonal to length & sentiment):
# length_dir then sentiment_dir, Gram-Schmidt, same order as
# layer_sweep.py's per-layer loop.
neutral_texts = LONG + SHORT + POS + NEG
neutral = read_many(neutral_texts, LAYER, "neutral(length/sentiment)")
length_dir = unit(neutral[:2].mean(0) - neutral[2:4].mean(0))
sentiment_dir = unit(neutral[4:6].mean(0) - neutral[6:8].mean(0))
assert len(LONG) == 2  # TODO: This is ugly, shouldn't hardcode assumed length
assert len(SHORT) == 2
assert len(POS) == 2
assert len(NEG) == 2

d = d_raw.copy()
confounds_removed = []
for name, cu in (("length", length_dir), ("sentiment", sentiment_dir)):
    d = d - np.dot(d, cu) * cu
    confounds_removed.append(name)
d = unit(d)

with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    held_out_separation = float(
        np.mean((chosen[val_idx] @ d) > (rejected[val_idx] @ d))
    )
print(
    f"held-out separation @ layer {LAYER}: {held_out_separation:.4f}  (shipped metadata: 1.0)"
)

d_f32 = d.astype(np.float32)

# %%
# ── Full validation + confound-audit suite (local, no generation-based
# steering check — see report note) ──────────────────────────────────────────
report: dict = {
    "model_id": MODEL_NAME,
    "layer": LAYER,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "local_dev": LOCAL_DEV,
}

SEP_MIN = 0.70
COS_MAX = 0.20

# 1. Held-out separation (already have it)
report["held_out_separation"] = round(held_out_separation, 4)

# 2. Confound cosines (validate_direction.py's proxy: same 2-text LONG/SHORT/POS/NEG)
cos_len = abs(float(np.dot(d, length_dir)))
cos_sent = abs(float(np.dot(d, sentiment_dir)))
report["cos_length"] = round(cos_len, 4)
report["cos_sentiment"] = round(cos_sent, 4)

# 3. Per-axis coherence
diffs = chosen - rejected
by_axis = defaultdict(list)
for i, r in enumerate(rows):
    by_axis[r["axis"]].append(diffs[i])
axis_cos = {ax: float(np.dot(unit(np.mean(v, axis=0)), d)) for ax, v in by_axis.items()}
coherence = float(np.mean(list(axis_cos.values())))
report["per_axis_coherence_mean"] = round(coherence, 4)
report["per_axis_cos"] = {k: round(v, 3) for k, v in sorted(axis_cos.items())}

g1 = held_out_separation >= SEP_MIN
g2 = cos_len < COS_MAX and cos_sent < COS_MAX
g3 = coherence > 0
passed = g1 and g2 and g3
report["auto_gates"] = {
    "held_out_separation": {
        "pass": g1,
        "value": held_out_separation,
        "need": f">= {SEP_MIN}",
    },
    "confound_cosines": {
        "pass": g2,
        "length": cos_len,
        "sentiment": cos_sent,
        "need": f"< {COS_MAX}",
    },
    "per_axis_coherence": {"pass": g3, "value": coherence, "need": "> 0"},
    "verdict": "PASS" if passed else "FAIL",
}

# Causal steering check: SKIPPED — needs autoregressive generation with the full
# model; validate_direction.py itself calls this one decisive only via human
# eyeball review. Not reproduced here; noted explicitly rather than faked.
report["causal_steering_check"] = (
    "SKIPPED — requires generation-based human review, not reproduced locally"
)

# Confound audit T3: estimator agreement (free — already-read activations)
meandiff_raw = unit(np.mean(chosen - rejected, axis=0))
X = np.vstack([chosen, rejected])
y = np.r_[np.ones(len(chosen)), np.zeros(len(rejected))]
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    lda_dir = unit(
        np.asarray(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            .fit(X, y)
            .coef_[0],
            dtype=np.float64,
        )
    )
    logit_dir = unit(
        np.asarray(
            LogisticRegression(C=0.1, max_iter=4000).fit(X, y).coef_[0],
            dtype=np.float64,
        )
    )

agreement = {
    "cos_meandiff_lda": round(float(np.dot(meandiff_raw, lda_dir)), 4),
    "cos_meandiff_logistic": round(float(np.dot(meandiff_raw, logit_dir)), 4),
    "cos_lda_logistic": round(float(np.dot(lda_dir, logit_dir)), 4),
    "cos_recovered_lda": round(float(np.dot(d, lda_dir)), 4),
    "cos_recovered_logistic": round(float(np.dot(d, logit_dir)), 4),
}
report["estimator_agreement"] = agreement

# Confound audit T1: independent confound-cosine battery
ap_act = read_many(APPROACH, LAYER, "approach")
av_act = read_many(AVOID, LAYER, "avoid")
pos_act = read_many(POS_AUDIT, LAYER, "pos(independent)")
neg_act = read_many(NEG_AUDIT, LAYER, "neg(independent)")
long_act = read_many(LONG, LAYER, "long(independent)")
short_act = read_many(SHORT, LAYER, "short(independent)")

approach_dir = unit(ap_act.mean(0) - av_act.mean(0))
valence_dir = unit(pos_act.mean(0) - neg_act.mean(0))
length_dir_indep = unit(long_act.mean(0) - short_act.mean(0))

with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    approach_probe = unit(
        np.asarray(
            LogisticRegression(C=0.1, max_iter=4000)
            .fit(
                np.vstack([ap_act, av_act]),
                np.r_[np.ones(len(ap_act)), np.zeros(len(av_act))],
            )
            .coef_[0],
            dtype=np.float64,
        )
    )

confound_flags = []
confound_cosines = {
    "cos_approach": round(abs(float(np.dot(d, approach_dir))), 4),
    "cos_valence": round(abs(float(np.dot(d, valence_dir))), 4),
    "cos_length_independent": round(abs(float(np.dot(d, length_dir_indep))), 4),
    "cos_approach_probe": round(abs(float(np.dot(d, approach_probe))), 4),
}
report["confound_cosines_independent"] = confound_cosines
for name, key in (
    ("approach/action", "cos_approach"),
    ("valence", "cos_valence"),
    ("approach-probe", "cos_approach_probe"),
):
    v = confound_cosines[key]
    if v >= 0.50:
        confound_flags.append(
            f"HIGH cos to {name} ({v:.2f}) — d is largely the {name} axis"
        )
    elif v >= 0.35:
        confound_flags.append(
            f"moderate cos to {name} ({v:.2f}) — partial {name} confound"
        )

# Confound audit T2: value-flip control pairs
kind_act = read_many([k for k, _ in CONTROL_PAIRS], LAYER, "control-kind")
cruel_act = read_many([c for _, c in CONTROL_PAIRS], LAYER, "control-cruel")
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    pk, pc = kind_act @ d, cruel_act @ d
    train_gap = float(np.mean((chosen @ d) - (rejected @ d)))
control_gap = float(np.mean(pk - pc))
frac_kind_higher = float(np.mean(pk > pc))
ratio = control_gap / train_gap if train_gap else float("nan")
control_pairs_report = {
    "frac_kind_projects_higher": round(frac_kind_higher, 3),
    "mean_control_gap": round(control_gap, 3),
    "mean_train_gap_chosen_rejected": round(train_gap, 3),
    "transfer_ratio": round(ratio, 3),
    "per_pair_gap": [round(g, 3) for g in (pk - pc).tolist()],
}
report["control_pairs"] = control_pairs_report
if frac_kind_higher < 0.70 or ratio < 0.30:
    confound_flags.append(
        f"control FAIL: kind>cruel only {frac_kind_higher:.0%} of the time, transfer ratio {ratio:.2f} "
        f"— d does NOT robustly prefer kind over cruel when both are active"
    )

report["confound_flags"] = confound_flags
report["confound_assessment"] = (
    "confound risk: see flags"
    if confound_flags
    else "no confound flag tripped by these tests"
)

# %%
# ── Step 5: save outputs, load shipped .npz READ-ONLY as comparison target ──
meta = {
    "model_id": MODEL_NAME,
    "layer": LAYER,
    "d_version": D_VERSION,
    "extraction_method": "logistic",
    "confounds_removed": confounds_removed,
    "held_out_separation": round(held_out_separation, 4),
    "num_pairs": N,
    "backend": "local",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "notes": "Recovered locally via recover_direction.py, reproducing steering-arena/scripts/layer_sweep.py's recipe (logistic, layer 24, length+sentiment orthogonalized) against locally computed activations (no NDIF).",
    "placeholder": False,
}
np.savez(OUT_NPZ, d=d_f32, meta=np.array(json.dumps(meta)))
print(f"wrote {OUT_NPZ}")

if not LOCAL_DEV and SHIPPED_NPZ.exists():
    shipped = np.load(SHIPPED_NPZ, allow_pickle=True)
    d_shipped = unit(np.asarray(shipped["d"], dtype=np.float64))
    shipped_meta = json.loads(str(shipped["meta"]))
    cos_vs_shipped = float(np.dot(d, d_shipped))

    shipped_audit = (
        json.loads(SHIPPED_AUDIT.read_text()) if SHIPPED_AUDIT.exists() else {}
    )

    report["reproduction_verdict"] = {
        "cosine_recovered_vs_shipped": round(cos_vs_shipped, 6),
        "shipped_meta": shipped_meta,
        "comparison": {
            "held_out_separation": {
                "shipped": shipped_meta.get("held_out_separation"),
                "recovered": round(held_out_separation, 4),
            },
            "cos_length": {"shipped": None, "recovered": cos_len},
            "cos_sentiment": {"shipped": None, "recovered": cos_sent},
            "estimator_agreement": {
                "shipped": shipped_audit.get("estimator_agreement"),
                "recovered": agreement,
            },
            "confound_cosines_independent": {
                "shipped": shipped_audit.get("confound_cosines"),
                "recovered": confound_cosines,
            },
            "control_pairs": {
                "shipped": shipped_audit.get("control_pairs"),
                "recovered": control_pairs_report,
            },
        },
    }
    print(f"\n=== REPRODUCTION VERDICT ===")
    print(f"cosine(recovered, shipped) = {cos_vs_shipped:.6f}")
elif not LOCAL_DEV:
    print(
        f"\nshipped comparison file not found at {SHIPPED_NPZ} — skipping reproduction verdict"
    )
else:
    print(
        "\n(LOCAL_DEV smoke test — skipping shipped-file comparison; run with LOCAL_DEV=False for the real recovery + verdict)"
    )

OUT_REPORT.write_text(json.dumps(report, indent=2))
print(f"wrote {OUT_REPORT}")
print(f"\nauto-gates verdict: {report['auto_gates']['verdict']}")
if confound_flags:
    print("confound flags:")
    for f in confound_flags:
        print(f"  ⚠ {f}")
else:
    print("no confound flags tripped")
