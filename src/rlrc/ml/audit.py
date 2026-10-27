"""Post-training edge cases; diagnostic evidence, never model-selection data."""

from pathlib import Path

from rlrc.config import load_settings
from rlrc.ml.benchmarks import ActorAdapter, controller_config
from rlrc.ml.evaluation import evaluate_episode, save_json
from rlrc.ml.runtime import NumpyActor
from rlrc.ml.simulator import SimConfig


def main() -> None:
    cfg = load_settings()
    actor = ActorAdapter(NumpyActor(Path("models/ppo-gateway"), controller_config(cfg, "rl", 24)))
    rows = []
    for name, demand, workers, service in (
        ("no-demand", 0, 4, 0.1),
        ("low-demand", 5, 4, 0.1),
        ("heavy-overload", 240, 4, 0.1),
        ("capacity-below-minimum-budget", 40, 1, 0.5),
    ):
        for policy in ("adaptive", "rl", "rl_safe"):
            row = evaluate_episode(
                SimConfig(
                    duration=90,
                    family="fixed",
                    fixed_offered=demand,
                    workers=workers,
                    service_time=service,
                    initial_rate=24,
                ),
                91000,
                "rl" if policy == "rl_safe" else policy,
                model=actor,
                safe=policy == "rl_safe",
                warmup=30,
            )
            row.update(
                case=name,
                label=policy,
                final_budget=row["budgets"][-1],
                peak_budget=max(row["budgets"]),
            )
            rows.append(
                {k: v for k, v in row.items() if k not in {"series", "budgets", "interventions"}}
            )
    save_json(Path("results/ml-runs/edge-audit.json"), rows)
    for row in rows:
        print(row["case"], row["label"], round(row["goodput_rps"], 2), row["final_budget"])


if __name__ == "__main__":
    main()
