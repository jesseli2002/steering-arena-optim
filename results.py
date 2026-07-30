"""Plot training results across different numbers of controlled tokens."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plot"

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


SCHEDULE_SWITCHES = [4, 8, 12]


def batch_size_for_iter(iter_idx, n_tokens):
    # Mirrors the BATCH_SIZE_OPTIM schedule in optimize_prompt.py.
    if iter_idx < n_tokens * 4:
        return 16
    elif iter_idx < n_tokens * 8:
        return 64
    elif iter_idx < n_tokens * 12:
        return 256
    else:
        return 1024


def plot_score_vs_iteration(histories, out_path):
    fig, ax = plt.subplots()
    for n_tokens in sorted(histories):
        records = histories[n_tokens]
        iters = [r["iter"] / n_tokens for r in records]
        scores = [r["score"] for r in records]
        ax.plot(iters, scores, label=f"{n_tokens} tokens")
    for x in SCHEDULE_SWITCHES:
        ax.axvline(x, color="grey", linestyle=":")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_xlabel("Iteration / number of controlled tokens")
    ax.set_ylabel("Training score")
    ax.set_title("Training score vs. iteration")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def cumulative_samples(records, n_tokens):
    cum_samples = []
    total = 0
    for r in records:
        total += batch_size_for_iter(r["iter"], n_tokens)
        cum_samples.append(total)
    return cum_samples


def plot_score_vs_samples(histories, out_path):
    FIT_TOKENS = [16, 32]

    fig, ax = plt.subplots()

    cum_samples_by_tokens = {}
    for n_tokens in sorted(histories):
        records = histories[n_tokens]
        cum_samples = cumulative_samples(records, n_tokens)
        cum_samples_by_tokens[n_tokens] = cum_samples
        scores = [r["score"] for r in records]
        ax.plot(cum_samples, scores, label=f"{n_tokens} tokens")

    fit_log_samples = np.concatenate(
        [np.log10(cum_samples_by_tokens[n]) for n in FIT_TOKENS]
    )
    fit_scores = np.concatenate(
        [[r["score"] for r in histories[n]] for n in FIT_TOKENS]
    )
    slope, intercept = np.polyfit(fit_log_samples, fit_scores, 1)
    fit_x = np.array([fit_log_samples.min(), fit_log_samples.max()])
    fit_label = (
        f"fit ({'+'.join(str(n) for n in FIT_TOKENS)} tokens)\nslope={slope:.4f}"
    )
    ax.plot(
        10**fit_x,
        slope * fit_x + intercept,
        color="grey",
        linestyle="--",
        label=fit_label,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative samples tried")
    ax.set_ylabel("Training score")
    ax.set_title("Training score vs. cumulative samples tried")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    histories = {n: load_history(run_dir) for n, run_dir in RUNS.items()}
    PLOT_DIR.mkdir(exist_ok=True)
    plot_max_score_vs_tokens(histories, PLOT_DIR / "max_score_vs_tokens.png")
    plot_score_vs_iteration(histories, PLOT_DIR / "score_vs_iteration.png")
    plot_score_vs_samples(histories, PLOT_DIR / "score_vs_samples.png")


if __name__ == "__main__":
    main()
