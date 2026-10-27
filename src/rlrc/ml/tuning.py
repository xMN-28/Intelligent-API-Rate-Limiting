"""A stronger conventional comparator, selected only on the fixed validation seeds."""

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlrc.ml.evaluation import evaluate_episode, save_json
from rlrc.ml.simulator import SimConfig


def tune() -> None:
    candidates: list[dict[str, Any]] = []
    for window, alpha, healthy, step in itertools.product([2, 5], [0.2, 0.5], [1, 3], [2, 5]):
        params = {
            "window": window,
            "alpha": alpha,
            "healthy_evaluations": healthy,
            "increase_step": step,
            "min_samples": 10 if window == 2 else 30,
        }
        episodes = [
            evaluate_episode(SimConfig(), seed, "adaptive", aimd_overrides=params)
            for seed in range(20001, 20007)
        ]
        score = float(
            np.mean(
                [
                    r["goodput_rps"]
                    - 10 * r["backend_error_fraction"]
                    - 5 * r["violation_seconds"] / r["duration"]
                    for r in episodes
                ]
            )
        )
        candidates.append(
            {
                "parameters": params,
                "validation_score": score,
                "validation_goodput": float(np.mean([r["goodput_rps"] for r in episodes])),
            }
        )
    best = max(candidates, key=lambda row: row["validation_score"])
    save_json(
        Path("models/aimd-tuned.json"),
        {
            "selected": best,
            "candidates": candidates,
            "seeds": list(range(20001, 20007)),
            "criterion": "Same independent validation score used for PPO checkpoint selection",
        },
    )
    print(best)


def heldout() -> None:
    from rlrc.ml.benchmarks import FAMILIES

    parameters = json.loads(Path("models/aimd-tuned.json").read_text())["selected"]["parameters"]
    output = Path("results/ml-runs/tuned-aimd-heldout")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for family in FAMILIES:
        for repeat in range(5):
            row = evaluate_episode(
                SimConfig(duration=180, family=family, initial_rate=24),
                40000 + repeat,
                "adaptive",
                warmup=30,
                aimd_overrides=parameters,
            )
            row["label"] = "adaptive_tuned"
            save_json(output / f"{family}-{repeat}.json", row)
            rows.append({k: v for k, v in row.items() if k not in {"series", "budgets"}})
    save_json(output / "summaries.json", rows)
    save_json(output / "selection.json", json.loads(Path("models/aimd-tuned.json").read_text()))
    print(float(np.mean([r["goodput_rps"] for r in rows])))


if __name__ == "__main__":
    import sys

    heldout() if "--heldout" in sys.argv else tune()
