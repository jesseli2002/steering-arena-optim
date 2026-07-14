import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

report_arena = json.loads(Path("data/arena_prefixes_result.json").read_text())
report_optim = json.loads(Path("data/score_check.json").read_text())

ground_truths = []
arena_scores = []
optim_scores = []

for result_arena, result_optim in zip(report_arena["results"], report_optim["results"]):
    prompt = result_arena["prompt"]
    assert prompt == result_optim["prompt"], "Prompt mismatches!"

    ground_truth = result_arena["ground_truth"]
    arena_score = result_arena["local_score"]
    optim_score = result_optim["score_estimate"]

    ground_truths.append(ground_truth)
    arena_scores.append(arena_score)
    optim_scores.append(optim_score)

ground_truth = np.array(ground_truth)
arena_scores = np.array(arena_scores)
optim_scores = np.array(optim_scores)

arena_diff = np.abs(arena_scores - ground_truth)
optim_diff = np.abs(optim_scores - arena_scores)

JITTER = 0.05
rng = np.random.default_rng()

fig, ax = plt.subplots()
N_DATA = len(arena_diff)
ax.scatter(
    np.full(N_DATA, 0) + JITTER * rng.uniform(-1, 1, size=N_DATA),
    arena_diff,
    label="local_score.py vs NDIF",
)
ax.scatter(
    np.full(N_DATA, 1) + JITTER * rng.uniform(-1, 1, size=N_DATA),
    optim_diff,
    label="GCG implementation vs local_score.py",
)
ax.legend()
ax.set_ylabel("Difference in mean score")
ax.set_yscale("log")
plt.show()
