"""Compare NDIF vs local (VastAI) OLMo-3 activation snapshots, layer by layer.

Loads two activation_snapshot*.npz files produced by
steering-arena/scripts/snapshot_activations.py (NDIF) and
snapshot_activations_local.py (VastAI), and for each (text, layer) pair
computes the Euclidean distance between the two last-token residual vectors.
Both tensors are upcast to float64 before any arithmetic, so the comparison
itself doesn't introduce float32 rounding as a confounder -- any remaining
floor in the distance reflects the float32 precision the snapshots were
saved at, not the diffing code.

This produces one plot, distance vs layer index, and nothing else. Reading
that plot is on the user: a floor near float32 machine epsilon (relative to
activation norm) at every layer points at a probe-direction/scoring bug
rather than an activation mismatch; growth with layer index points at
numerical drift (dtype, kernels, sharding) accumulating through the model.
This script does not decide which -- that's inherently a judgment call about
what "near epsilon" or "growing" means for a given run, and hardcoding a
threshold here would just hide a different failure mode if one occurs.

    python compare_activations.py
    python compare_activations.py --ndif <path> --local <path> --out <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
ARENA_ROOT = REPO_ROOT / "steering-arena"

MEAN_COLOR = "#2a78d6"  # dataviz categorical slot 1
TRACE_COLOR = "0.75"  # recessive gray for per-text context lines


def _latest(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    return matches[-1]


def load_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (acts, layers, texts): acts is (n_texts, n_layers, d_model) float64."""
    data = np.load(path, allow_pickle=True)
    layers = np.asarray(data["layers"])
    texts = np.asarray(data["texts"])
    acts = np.stack(
        [np.asarray(data[f"acts_{i}"], dtype=np.float64) for i in range(len(texts))]
    )
    return acts, layers, texts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ndif",
        type=Path,
        default=None,
        help="NDIF activation_snapshot_<date>.npz (default: latest under "
        "steering-arena/data/analysis/)",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help="Local/VastAI activation_snapshot_local_<date>.npz (default: latest "
        "under data/analysis/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "analysis" / "activation_diff.png",
        help="Output plot path (default: data/analysis/activation_diff.png)",
    )
    args = parser.parse_args()

    ndif_path = args.ndif or _latest(
        "activation_snapshot_[0-9]*.npz", ARENA_ROOT / "data" / "analysis"
    )
    local_path = args.local or _latest(
        "activation_snapshot_local_*.npz", REPO_ROOT / "data" / "analysis"
    )

    ndif_acts, ndif_layers, ndif_texts = load_snapshot(ndif_path)
    local_acts, local_layers, local_texts = load_snapshot(local_path)

    assert list(ndif_texts) == list(
        local_texts
    ), "text sets differ between snapshots -- they must cover the same fixed texts to be comparable"
    assert list(ndif_layers) == list(
        local_layers
    ), f"layer sets differ: ndif={list(ndif_layers)} local={list(local_layers)}"
    layers = ndif_layers

    dist = np.linalg.norm(ndif_acts - local_acts, axis=-1)  # (n_texts, n_layers)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in dist:
        ax.plot(layers, row, color=TRACE_COLOR, linewidth=0.8, zorder=1)
    ax.plot(
        layers,
        dist.mean(axis=0),
        color=MEAN_COLOR,
        linewidth=2.5,
        marker="o",
        label="mean over texts",
        zorder=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("layer index")
    ax.set_ylabel("Euclidean distance (float64)")
    ax.set_title("NDIF vs local activation distance by layer")
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    print(f"ndif snapshot:  {ndif_path}")
    print(f"local snapshot: {local_path}")
    print(f"wrote {args.out}\n")
    print(f"{'layer':>6}  {'mean':>12}  {'median':>12}  {'max':>12}")
    for layer, row in zip(layers, dist.T):
        print(
            f"{layer:6d}  {row.mean():12.6e}  {np.median(row):12.6e}  {row.max():12.6e}"
        )


if __name__ == "__main__":
    main()
