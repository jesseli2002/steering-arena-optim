"""Plot training results across different numbers of controlled tokens."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

RUNS = {
    8: "2026-07-28_tok8_pro",
    16: "2026-07-28_tok16_pro",
    32: "2026-07-28_tok32_pro",
}


def load_history(run_dir):
    history_path = RESULTS_DIR / run_dir / "history.jsonl"
    with open(history_path) as f:
        return [json.loads(line) for line in f]


def plot_max_score_vs_tokens(histories, out_path):
    fig, ax = plt.subplots()
    n_tokens = sorted(histories)
    max_scores = [max(r["score"] for r in histories[n]) for n in n_tokens]
    ax.plot(n_tokens, max_scores, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_tokens)
    ax.set_xticklabels(n_tokens)
    ax.set_xlabel("Number of controlled tokens")
    ax.set_ylabel("Max score")
    ax.set_title("Max score vs. number of controlled tokens")
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_vs_iteration(histories, out_path):
    fig, ax = plt.subplots()
    for n_tokens in sorted(histories):
        records = histories[n_tokens]
        iters = [r["iter"] for r in records]
        scores = [r["score"] for r in records]
        ax.plot(iters, scores, label=f"{n_tokens} tokens")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training score")
    ax.set_title("Training score vs. iteration")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    histories = {n: load_history(run_dir) for n, run_dir in RUNS.items()}
    plot_max_score_vs_tokens(histories, RESULTS_DIR / "max_score_vs_tokens.png")
    plot_score_vs_iteration(histories, RESULTS_DIR / "score_vs_iteration.png")


if __name__ == "__main__":
    main()
