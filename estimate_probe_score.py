# %%
# Estimate the mean Season 2 probe score for a user-provided steering prompt,
# using the direction recovered by recover_direction.py and the 16 prompts in
# steering-arena/data/probes/season2.json.
#
# Composition and scoring math (compose/cosine/load_direction/load_probes)
# are copied verbatim from the read-only steering-arena/app/scoring.py, same
# convention as recover_direction.py copying from extract_direction.py, for
# exact parity with the live scorer without importing arena code:
#   score(seq) = mean over probes p of
#       cos(R_L(seq + " " + p)[-1], d) - cos(R_L(p)[-1], d)
# where seq+" "+p is compose(seq, p) (plain space-joined concatenation, no
# chat template) and R_L(...)[-1] is the last-token layer-L residual stream.
# See app/scoring.py and PROJECT_SPEC.md Section 5.
#
# `steering-arena/` is read-only reference (CLAUDE.md) — never written to,
# never imported from.
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch as t
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
ARENA_ROOT = REPO_ROOT / "steering-arena"

if not ARENA_ROOT.is_dir() and os.environ.get("STEERING_ARENA_DIR"):
    ARENA_ROOT = Path(os.environ["STEERING_ARENA_DIR"]).resolve()
assert (
    ARENA_ROOT.is_dir()
), f"expected read-only steering-arena clone at {ARENA_ROOT} (or set STEERING_ARENA_DIR)"


# %%
# Copy implementations from steering-arena/app/scoring.py
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


def load_direction(path) -> tuple[np.ndarray, dict]:
    """Load `d` (float32) and its metadata from a d_<version>.npz file."""
    data = np.load(path, allow_pickle=True)
    d = np.asarray(data["d"], dtype=np.float32)
    meta = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"]))
    return d, meta


def load_prompt_prefixes(path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["prompts"] if isinstance(obj, dict) else list(obj)


# %%
# Real recovered direction lives under data/; the smoke direction under
# data_local/ (machine-local scratch), matching recover_direction.py's tiers.
DEFAULT_DIRECTION_CANDIDATES = [
    REPO_ROOT / "data" / "directions" / "d_olmo3_L24_logistic.recovered.npz",
    REPO_ROOT / "data_local" / "directions" / "d_dev_smoke.npz",
]
DEFAULT_SEASON_FILE = ARENA_ROOT / "data" / "probes" / "season2.json"


def pick_default_direction() -> Path:
    for p in DEFAULT_DIRECTION_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "no direction found under data/directions/ or data_local/directions/ — run "
        "recover_direction.py first, or pass --direction explicitly"
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactively estimate the mean Season 2 probe score for "
        "steering prompts, loading the model once and looping over stdin."
    )
    ap.add_argument(
        "--direction",
        type=Path,
        default=None,
        help="path to a d_*.npz direction file (default: auto-detect under data/directions/)",
    )
    ap.add_argument(
        "--season-file",
        type=Path,
        default=DEFAULT_SEASON_FILE,
        help="path to a season probes json (default: steering-arena's season2.json)",
    )
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    return ap.parse_args(argv)


# %%
args = parse_args()

direction_path = args.direction or pick_default_direction()
d, meta = load_direction(direction_path)
prefixes = load_prompt_prefixes(args.season_file)

print(f"direction: {direction_path}")
print(
    f"  model_id={meta.get('model_id')} layer={meta.get('layer')} placeholder={meta.get('placeholder')}"
)
print(f"prefixes: {len(prefixes)} from {args.season_file}")
if meta.get("placeholder"):
    print("[warning] using a PLACEHOLDER direction -- scores are not meaningful yet.")

MODEL_NAME = meta["model_id"]
LAYER = int(meta["layer"])

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16 if device.type == "cuda" else t.float32
if device.type != "cuda":
    print("\033[93mWarning: CUDA not available, using CPU. This will be slow.\033[0m")
if t.cuda.is_available():
    t.cuda.set_device(0)

HF_TOKEN = os.getenv(args.hf_token_env)
assert HF_TOKEN, f"Please set {args.hf_token_env}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=dtype, device_map="auto", token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

# %%
# Disk-cached last-token layer-L residual reads, one text at a time (no
# padding), same convention verified in recover_direction.py: HF's
# hidden_states[i+1] is decoder layer i's output, i.e. hidden_states[layer+1]
# == model.model.layers[layer].output, matching the last-token read the live
# scorer performs via NDIF/nnsight.
CACHE_DIR = REPO_ROOT / ".cache" / "acts" / MODEL_NAME.replace("/", "_")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_fp(text: str, layer: int) -> Path:
    key = hashlib.sha256(
        f"{MODEL_NAME}\x00L{layer}\x00{text}".encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / f"{key}.npy"


@t.no_grad()
def last_token_resid(text: str, layer: int) -> np.ndarray:
    fp = _cache_fp(text, layer)
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


def get_resid(text: str) -> np.ndarray:
    return last_token_resid(text, LAYER)


# %%
# Per-probe shift, mirroring app.scoring.steering_shift_score's loop exactly
# so the printed breakdown and the mean are computed by the identical
# formula the live scorer uses.
def score_prompt(prompt: str) -> None:
    shifts = []
    for p in tqdm(prefixes, desc="scoring prefixes"):
        with_seq = cosine(get_resid(compose(prompt, p)), d)
        base = cosine(get_resid(p), d)
        shift = with_seq - base
        shifts.append(shift)
        print(f"  {shift:+.4f}  {p!r}")

    mean_score = float(np.mean(shifts))
    print(
        f"\nmean probe score (cosine_steering_shift) over {len(prefixes)} tests: {mean_score:.4f}"
    )

    # Score lands in the same tier as the direction it was computed from: a
    # smoke direction (data_local/) yields a smoke score, a real one (data/)
    # a real score.
    artefact_root = REPO_ROOT / (
        "data_local" if "data_local" in direction_path.resolve().parts else "data"
    )
    out_dir = artefact_root / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"score_{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "direction": str(direction_path),
                "direction_meta": meta,
                "season_file": str(args.season_file),
                "per_probe_shift": dict(zip(prefixes, shifts)),
                "mean_score": mean_score,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


def load_new_direction(path_str: str) -> None:
    global direction_path, d, meta, LAYER, MODEL_NAME, tokenizer, model, CACHE_DIR

    new_path = Path(path_str).expanduser()
    if not new_path.exists():
        print(f"[error] direction file not found: {new_path}")
        return
    new_d, new_meta = load_direction(new_path)
    new_model_name = new_meta["model_id"]

    if new_model_name != MODEL_NAME:
        print(f"[reloading model] {MODEL_NAME} -> {new_model_name}")
        MODEL_NAME = new_model_name
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=dtype, device_map="auto", token=HF_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
        CACHE_DIR = REPO_ROOT / ".cache" / "acts" / MODEL_NAME.replace("/", "_")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    direction_path, d, meta = new_path, new_d, new_meta
    LAYER = int(meta["layer"])

    print(f"direction: {direction_path}")
    print(
        f"  model_id={meta.get('model_id')} layer={meta.get('layer')} placeholder={meta.get('placeholder')}"
    )
    if meta.get("placeholder"):
        print("[warning] using a PLACEHOLDER direction -- scores are not meaningful yet.")


# %%
# Interactive loop: the model and prefixes stay loaded across iterations, so
# only the first response (what kind of input) and second response (the
# input itself) are prompted for each round.
print("\nready. Enter prompts to score, or swap in a new direction file.")
while True:
    kind = input("\n[p]rompt / [d]irection file / [q]uit? ").strip().lower()
    if kind in ("q", "quit", "exit"):
        break
    elif kind in ("p", "prompt"):
        prompt = input("prompt: ").strip()
        if not prompt:
            print("[error] empty prompt, skipping")
            continue
        score_prompt(prompt)
    elif kind in ("d", "direction"):
        path_str = input("direction file path: ").strip()
        if not path_str:
            print("[error] empty path, skipping")
            continue
        load_new_direction(path_str)
    else:
        print(f"[error] unrecognized choice {kind!r}, enter p/d/q")
