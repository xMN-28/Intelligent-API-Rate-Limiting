"""Evidence-backed figures and Markdown; simulator and HTTP populations stay separate."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from rlrc.ml.evaluation import save_json
from rlrc.report import summarize

LABELS = {
    "none": "No limiter",
    "token_bucket": "Token Bucket · low",
    "static_high": "Token Bucket · high",
    "sliding_window": "Sliding Window · low",
    "sliding_high": "Sliding Window · high",
    "adaptive": "EWMA + AIMD",
    "rl": "PPO · gateway",
    "rl_safe": "PPO + safety",
    "rl_backend": "PPO · backend telemetry",
    "predictive": "Prediction + AIMD",
    "hybrid": "PPO + prediction + safety",
    "rl_prediction": "PPO + prediction",
    "adaptive_tuned": "AIMD · validation-tuned",
}
COLORS = {
    "none": "#8a929b",
    "token_bucket": "#7597a7",
    "static_high": "#456d82",
    "sliding_window": "#7597a7",
    "sliding_high": "#456d82",
    "adaptive": "#243f56",
    "rl": "#b97d37",
    "rl_safe": "#885c2d",
    "rl_backend": "#d0aa6d",
    "predictive": "#7b844b",
    "hybrid": "#9c5e70",
    "rl_prediction": "#ba8793",
    "adaptive_tuned": "#387e87",
}
ORDER = [
    "none",
    "token_bucket",
    "static_high",
    "sliding_window",
    "sliding_high",
    "adaptive",
    "adaptive_tuned",
    "predictive",
    "rl",
    "rl_safe",
    "rl_backend",
    "rl_prediction",
    "hybrid",
]


def interval(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return float("nan"), None
    mean = float(np.mean(values))
    error = (
        float(student_t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else None
    )
    return mean, error


def practical_findings(groups: dict[str, list[dict[str, Any]]], complete: bool) -> list[str]:
    if not complete:
        return ["Final interpretation is pending the scheduled repetitions."]
    best = max(groups, key=lambda name: np.mean([r["goodput_rps"] for r in groups[name]]))
    lines = [
        f"The highest observed mean goodput is {LABELS[best]}. This ranking is descriptive; "
        "three workload repetitions do not establish a universal best controller."
    ]
    policy = {r["seed"]: r for r in groups.get("rl_safe", [])}
    for name in ("adaptive", "static_high"):
        diffs = [
            policy[r["seed"]]["goodput_rps"] - r["goodput_rps"]
            for r in groups.get(name, [])
            if r["seed"] in policy
        ]
        if len(diffs) < 2:
            continue
        mean, error = interval(diffs)
        assert error is not None
        interpretation = (
            "The descriptive interval excludes zero."
            if mean - error > 0 or mean + error < 0
            else "The interval includes zero; the goodput difference is inconclusive."
        )
        lines.append(
            f"PPO + safety minus {LABELS[name]}: {mean:+.2f} ± {error:.2f} RPS "
            f"(paired 95% t interval). {interpretation} These intervals are not "
            "adjusted for multiple comparisons."
        )
    lines.append(
        "Read goodput together with backend errors and tail latency. Restoring capacity "
        "does not erase queued work during abrupt degradation. "
        "The shared minimum budget and more frequent PPO action changes remain "
        "limitations. Neither forecast accuracy nor simulator goodput alone proves "
        "a deployment benefit."
    )
    for name in ("adaptive", "rl_safe"):
        group = groups.get(name, [])
        recoveries = [r["recovery_seconds"] for r in group if r["recovery_seconds"] is not None]
        if recoveries:
            lines.append(
                f"{LABELS[name]} relative recovery averaged {np.mean(recoveries):.1f}s "
                f"among {len(recoveries)}/{len(group)} recovered trials; mean absolute "
                f"log budget change was {np.mean([r['oscillation'] for r in group]):.3f}. "
                "The preserved recovery metric references each controller's own "
                "pre-degradation throughput. PPO's goodput gain did not also establish "
                "a recovery or oscillation improvement."
            )
    return lines


def finish(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def dynamics_figures(real_root: Path, output: Path) -> None:
    """Preselected seed 81000, including failures; never select a favorable trace."""
    names = ["static_high", "adaptive", "rl", "rl_safe", "hybrid"]
    runs = {}
    for path in real_root.glob("run-*/experiment.json"):
        item = json.loads(path.read_text())
        if item["seed"] == 81000 and item["label"] in names:
            runs[item["label"]] = path.parent
    if not runs:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    traffic, panels = plt.subplots(
        len(runs), 1, figsize=(12, 3 * len(runs)), sharex=True, squeeze=False, layout="constrained"
    )
    for index, (name, run) in enumerate(sorted(runs.items(), key=lambda r: names.index(r[0]))):
        manifest = json.loads((run / "manifest.json").read_text())
        duration = sum(p["duration"] for p in manifest["scenario"]["phases"])
        origin = manifest["measured_start_wall"]
        with (run / "requests.csv").open(newline="") as file:
            rows = [r for r in csv.DictReader(file) if 0 <= float(r["offset"]) < duration]
        decisions_path = run / "decisions.jsonl"
        decisions = (
            [json.loads(line) for line in decisions_path.read_text().splitlines()]
            if decisions_path.exists()
            else []
        )
        applied = [d for d in decisions if d["applied"]]
        if applied:
            axes[0].step(
                [d["wall_time"] - origin for d in applied],
                [d["rate"] for d in applied],
                where="post",
                label=LABELS[name],
                color=COLORS[name],
            )
        else:
            axes[0].axhline(manifest["settings"]["rate"], label=LABELS[name], color=COLORS[name])
        edges = np.arange(0, duration + 1, 1)
        values: list[list[float]] = [[] for _ in range(int(duration))]
        for row in rows:
            if row["outcome"] == "success":
                values[int(float(row["offset"]))].append(float(row["latency"]))
        axes[1].plot(
            edges[:-1],
            [np.quantile(v, 0.95) * 1000 if v else np.nan for v in values],
            label=LABELS[name],
            color=COLORS[name],
            alpha=0.8,
        )
        ax = panels[index, 0]
        for label, subset, color in (
            ("Offered", rows, "#8a929b"),
            ("Admitted", [r for r in rows if r["admitted"] == "True"], "#7597a7"),
            ("Successful", [r for r in rows if r["outcome"] == "success"], "#b97d37"),
            (
                "Goodput",
                [
                    r
                    for r in rows
                    if r["outcome"] == "success"
                    and float(r["latency"]) <= manifest["settings"]["controller"]["latency_target"]
                ],
                "#243f56",
            ),
        ):
            counts, _ = np.histogram([float(r["offset"]) for r in subset], bins=edges)
            ax.step(edges[:-1], counts, where="post", label=label, color=color)
        ax.set(ylabel="Requests/s", title=LABELS[name], ylim=(0, 70))
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=0.2)
        ax.axvspan(20, 40, color="#9c5e70", alpha=0.1)
    axes[0].set_ylabel("Global budget (RPS)")
    axes[1].set_ylabel("Successful p95 (ms)")
    axes[1].axhline(300, color="#7b3434", linestyle="--", label="Latency target")
    for ax in axes:
        ax.axvspan(20, 40, color="#9c5e70", alpha=0.1)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8)
    axes[1].set_xlabel("Measured workload time (seconds)")
    panels[-1, 0].set_xlabel("Measured workload time (seconds)")
    fig.suptitle("Capacity recovery · fixed first seed 81000 · shaded = one worker")
    traffic.suptitle("Request accounting · fixed first seed 81000 · shaded = one worker")
    finish(fig, output / "real-dynamics.png")
    finish(traffic, output / "real-traffic.png")


def comparison(rows: list[dict[str, Any]], output: Path, title: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    names = [name for name in ORDER if name in groups]
    if not names:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, max(8, len(names) * 0.65)), layout="constrained")
    for axis, metric, label, multiplier in zip(
        axes.flat,
        ("goodput_rps", "latency_p99", "backend_error_fraction", "rejection_fraction"),
        (
            "Goodput (RPS)",
            "Successful-response p99 (ms)",
            "Backend errors / offered (%)",
            "Gateway rejections / offered (%)",
        ),
        (1, 1000, 100, 100),
        strict=True,
    ):
        points = []
        for name in names:
            values = [r[metric] * multiplier for r in groups[name] if r.get(metric) is not None]
            mean, error = interval(values)
            points.append(mean)
            index = names.index(name)
            axis.barh(index, mean, color=COLORS[name], alpha=0.95)
            if error is not None:
                axis.errorbar(mean, index, xerr=error, fmt="none", color="#23313d", capsize=3)
        axis.set_yticks(range(len(names)), [LABELS[n] for n in names], fontsize=9)
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.set_xlim(left=0)
        axis.grid(axis="x", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        title + "\nMeans and 95% Student-t intervals across independent seeds", fontsize=14
    )
    finish(fig, output)


def training_figures(output: Path) -> None:
    models = [
        ("ppo-gateway", "PPO gateway", "#b97d37"),
        ("ppo-backend-v2", "PPO backend", "#456d82"),
        ("ppo-prediction-v2", "PPO + prediction", "#9c5e70"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for directory, label, color in models:
        rows = json.loads((Path("models") / directory / "validation.json").read_text())
        xs = [r["steps"] / 1000 for r in rows]
        axes[0].plot(
            xs,
            [sum(r["reward_components"].values()) for r in rows],
            marker="o",
            label=label,
            color=color,
        )
        axes[1].plot(xs, [r["goodput"] for r in rows], marker="o", label=label, color=color)
    axes[0].set_ylabel("Mean training reward / transition")
    axes[1].set_ylabel("Validation goodput (RPS)")
    for ax in axes:
        ax.set_xlabel("Training transitions (thousands)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("Training progress · fixed validation scenarios, not held-out tests")
    finish(fig, output / "training.png")
    rows = json.loads(Path("models/ppo-gateway/validation.json").read_text())
    components = list(rows[0]["reward_components"])
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    colors = ["#456d82", "#9c5e70", "#885c2d", "#7b844b", "#8a929b", "#b97d37"]
    for key, color in zip(components, colors, strict=True):
        ax.plot(
            [r["steps"] / 1000 for r in rows],
            [r["reward_components"][key] for r in rows],
            label=key,
            color=color,
            marker="o",
        )
    ax.axhline(0, color="#23313d", linewidth=0.8)
    ax.set(
        xlabel="Training transitions (thousands)",
        ylabel="Weighted reward component",
        title="PPO gateway · reward components over training windows",
    )
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    finish(fig, output / "reward-components.png")


def hardware_figure(output: Path) -> None:
    data = json.loads(Path("results/ml-runs/hardware.json").read_text())
    labels = ["1 environment", "4 environments", "4 subprocesses"]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for index, device in enumerate(("cpu", "cuda")):
        rows = [r for r in data["results"] if r["device"] == device]
        if rows:
            ax.bar(
                np.arange(3) + (index - 0.5) * 0.35,
                [r["steps_s"] for r in rows],
                width=0.35,
                label=device.upper(),
                color=("#456d82", "#b97d37")[index],
            )
    ax.set_xticks(range(3), labels)
    ax.set(
        ylabel="PPO training transitions / second",
        title="Measured CPU/CUDA probe · small 64×64 MLP",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    finish(fig, output / "hardware.png")


def fidelity_figures(fidelity_root: Path, output: Path) -> None:
    rows = json.loads((fidelity_root / "steady-fidelity.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    offered = [r["offered"] for r in rows]
    for axis, real_key, sim_key, label in zip(
        axes,
        ("success_rps", "latency_p95", "queue"),
        ("sim_success_rps", "sim_p95", "sim_mean_queue"),
        ("Successful throughput (RPS)", "Successful p95 (seconds)", "Mean backend queue"),
        strict=True,
    ):
        measured = [
            r["real_mean_queue"] if real_key == "queue" else r["real"][real_key] for r in rows
        ]
        axis.plot(offered, measured, "o-", label="Real HTTP", color="#243f56")
        axis.plot(
            offered, [r[sim_key] for r in rows], "s--", label="Event simulator", color="#b97d37"
        )
        axis.set(xlabel="Offered load (RPS)", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(
        "Simulator fidelity · unthrottled, fixed backend; nominal parameters, no final-test fitting"
    )
    finish(fig, output / "fidelity.png")
    with (fidelity_root / "dynamic" / "requests.csv").open() as file:
        real = [r for r in csv.DictReader(file) if float(r["offset"]) >= 0]
    simulation = json.loads((fidelity_root / "dynamic-simulation.json").read_text())[10:]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for key, axis in zip(("success", "backend_error"), axes, strict=True):
        field = "success" if key == "success" else "backend_errors"
        real_values = [
            sum(r["outcome"] == key and second <= float(r["offset"]) < second + 5 for r in real) / 5
            for second in range(0, 70, 5)
        ]
        sim_values = [
            sum(row[field] for row in simulation[second : second + 5]) / 5
            for second in range(0, 70, 5)
        ]
        axis.step(range(0, 70, 5), real_values, where="post", label="Real HTTP", color="#243f56")
        axis.step(
            range(0, 70, 5),
            sim_values,
            where="post",
            label="Simulator",
            color="#b97d37",
            linestyle="--",
        )
        axis.axvspan(20, 40, alpha=0.12, color="#8a929b")
        axis.set_ylabel("Success RPS" if key == "success" else "Backend errors / s")
        axis.legend()
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Measured time (seconds); shaded: one worker")
    fig.suptitle("Capacity drop and recovery · unthrottled fidelity experiment")
    finish(fig, output / "fidelity-dynamic.png")


def simulator_figure(rows: list[dict[str, Any]], output: Path) -> list[str]:
    families = list(dict.fromkeys(r["family"] for r in rows))
    names = [name for name in ORDER if any(r["label"] == name for r in rows)]
    matrix = np.asarray(
        [
            [
                np.mean(
                    [r["goodput_rps"] for r in rows if r["label"] == name and r["family"] == family]
                )
                for family in families
            ]
            for name in names
        ]
    )
    fig, ax = plt.subplots(figsize=(14, 7.5), layout="constrained")
    heat = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=np.ceil(matrix.max()), aspect="auto")
    ax.set_xticks(range(len(families)), families, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    for i in range(len(names)):
        for j in range(len(families)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] > matrix.max() * 0.6 else "#23313d",
            )
    ax.set_title("Held-out simulator goodput · mean RPS, five matched seeds per family")
    fig.colorbar(heat, ax=ax, label="Goodput (RPS)")
    finish(fig, output / "simulator.png")
    winners = []
    for j, family in enumerate(families):
        name = names[int(np.argmax(matrix[:, j]))]
        winners.append(f"| {family} | {LABELS[name]} | {matrix[:, j].max():.2f} |")
    return winners


def risk_figure(output: Path) -> None:
    with np.load("models/overload-predictor/heldout-predictions.npz", allow_pickle=False) as values:
        ys, probabilities = values["y"], values["probability"]
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    xs, observed = [], []
    for low, high in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1), strict=True):
        mask = (probabilities >= low) & (probabilities < high + 0.00000001)
        if mask.any():
            xs.append(probabilities[mask].mean())
            observed.append(ys[mask].mean())
    ax.plot([0, 1], [0, 1], "--", color="#8a929b", label="Perfect calibration")
    ax.plot(xs, observed, "o-", color="#7b844b", label="3-second risk model")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed risk fraction",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Scenario-held-out predictor calibration",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    finish(fig, output / "prediction.png")


def intervention_figure(rows: list[dict[str, Any]], output: Path) -> None:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for reason, count in row["safety_overrides"].items():
            groups[row["label"]][reason] += count
    if not groups:
        return
    names = [name for name in ORDER if name in groups]
    reasons = sorted({reason for counts in groups.values() for reason in counts})
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    left = np.zeros(len(names))
    for reason in reasons:
        counts = np.asarray([groups[name].get(reason, 0) for name in names])
        ax.barh(range(len(names)), counts, left=left, label=reason)
        left += counts
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax.set(
        xlabel="Decisions changing the model proposal (count across measured trials)",
        title="Explicit safety overrides · overlapping health conditions log one reason",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    finish(fig, output / "safety.png")


def build_upgrade_report(
    real_root: Path, sim_root: Path, fidelity_root: Path, output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    sim = json.loads((sim_root / "summaries.json").read_text())
    real = []
    for path in sorted(real_root.glob("run-*/manifest.json")):
        manifest = json.loads(path.read_text())
        if not manifest.get("complete") or not (path.parent / "experiment.json").exists():
            continue
        row = summarize(path.parent)
        item = json.loads((path.parent / "experiment.json").read_text())
        row.update(label=item["label"], seed=item["seed"])
        real.append(row)
    valid = [r for r in real if r["valid_generator"] and r["valid_controller"]]
    tuned_path = Path("results/ml-runs/tuned-aimd-heldout/summaries.json")
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else []
    edge_path = Path("results/ml-runs/edge-audit.json")
    edges = json.loads(edge_path.read_text()) if edge_path.exists() else []
    predictor_metrics = json.loads(Path("models/overload-predictor/metadata.json").read_text())
    supplemental = []
    for path in Path("results/ml-runs/tuned-aimd-real").glob("run-*/experiment.json"):
        row = summarize(path.parent)
        if row["complete"] and row["valid_generator"] and row["valid_controller"]:
            supplemental.append(row)
    supplemental_mean, supplemental_error = interval([r["goodput_rps"] for r in supplemental])
    supplemental_result = (
        f"Supplemental tuned-AIMD mean goodput: {supplemental_mean:.2f} ± "
        f"{supplemental_error:.2f} RPS (95% t interval)."
        if supplemental_error is not None
        else "Supplemental aggregate awaits multiple trials."
    )
    protocol = json.loads((real_root / "protocol.json").read_text())
    expected = len(protocol["schedule"])
    complete = len(real) == expected
    training_figures(output)
    hardware_figure(output)
    fidelity_figures(fidelity_root, output)
    risk_figure(output)
    winners = simulator_figure(sim, output)
    comparison(
        valid, output / "real-comparison.png", "Real HTTP/Redis · capacity drop and recovery"
    )
    dynamics_figures(real_root, output)
    intervention_figure(valid, output)
    summary_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        summary_groups[row["label"]].append(row)
    summary_groups = {name: summary_groups[name] for name in ORDER if name in summary_groups}
    lines = [
        "# Reinforcement-learning upgrade: measured results",
        "",
        f"HTTP evaluation status: **{'COMPLETE' if complete else 'IN PROGRESS'} — "
        f"{len(real)}/{expected} trials finished**, {len(valid)} valid.",
        "",
        "Simulator results and real HTTP results below are distinct evidence. "
        "Models were frozen before final comparisons. These local experiments do not "
        "establish production readiness or research novelty.",
        "",
        "## Real HTTP/Redis comparison",
        "",
        *[paragraph + "\n" for paragraph in practical_findings(summary_groups, complete)],
        "",
        "![HTTP comparisons](real-comparison.png)",
        "",
        "Three matched workload seeds were scheduled per controller, randomized within "
        "each repetition. Each trial has 30 seconds warm-up and 100 seconds measurement: "
        "20 seconds normal, 20 seconds degraded, 60 seconds restored. Offered load is "
        "60 RPS throughout; workers change 4→1→4 with 100ms nominal service and an "
        "eight-request backend queue. The same timeout, fairness allocation, target, "
        "gateway ceiling, and forwarding path apply to every controller. "
        "This is a focused local study, not the full 300-second multi-workload HTTP matrix.",
        "",
        "| Controller | Valid n | Goodput RPS (95% t interval) | Successful p99 ms | "
        "Backend errors % offered | Rejections % offered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, group in summary_groups.items():
        mean, error = interval([r["goodput_rps"] for r in group])
        precision = f"{mean:.2f} ± {error:.2f}" if error is not None else f"{mean:.2f} (single)"
        tails = [r["latency_p99"] * 1000 for r in group if r["latency_p99"] is not None]
        lines.append(
            f"| {LABELS[name]} | {len(group)} | {precision} | {np.mean(tails):.1f} | "
            f"{np.mean([r['backend_error_fraction'] for r in group]) * 100:.2f} | "
            f"{np.mean([r['rejection_fraction'] for r in group]) * 100:.2f} |"
        )
    lines += [
        "",
        "| Controller | Success RPS | p50 ms | p95 ms | Timeouts % offered | "
        "Offered / launched (total) | Missed launches % | Model fallback decisions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in summary_groups.items():
        lines.append(
            f"| {LABELS[name]} | {np.mean([r['success_rps'] for r in group]):.2f} | "
            f"{np.mean([r['latency_p50'] for r in group]) * 1000:.1f} | "
            f"{np.mean([r['latency_p95'] for r in group]) * 1000:.1f} | "
            f"{np.mean([r['timeout_fraction'] for r in group]) * 100:.3f} | "
            f"{sum(r['offered'] for r in group)} / {sum(r['launched'] for r in group)} | "
            f"{np.mean([r['missed_launch_fraction'] for r in group]) * 100:.3f} | "
            f"{sum(r['fallback_count'] for r in group)} |"
        )
    lines += [
        "",
        "### Recovery, stability, and overhead",
        "",
        "| Controller | Recovery s | Recovered n/total | SLO-violating seconds | Jain fairness | "
        "Mean absolute log budget change | Inference p95 ms | Control p95 ms | "
        "Gateway RSS p95 MB | Gateway CPU p95 % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in summary_groups.items():
        values = []
        for key in (
            "recovery_seconds",
            "violation_seconds",
            "jain_fairness",
            "oscillation",
            "inference_p95_ms",
            "controller_p95_ms",
            "rss_p95_mb",
            "cpu_p95_percent",
        ):
            observed = [r[key] for r in group if r.get(key) is not None]
            values.append(f"{np.mean(observed):.3f}" if observed else "—")
        recovered = sum(r["recovery_seconds"] is not None for r in group)
        values.insert(1, f"{recovered}/{len(group)}")
        lines.append(f"| {LABELS[name]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Recovery requires three consecutive 5s windows reaching 80% of that run's "
        "pre-degradation goodput with successful p95 within target. It measures stability "
        "relative to each controller's own baseline, not equal absolute capacity. Missing "
        "recoveries are censored; consult individual summaries. SLO-violating seconds "
        "count arrival cohorts with successful p95 above target. CPU/RSS are whole-gateway "
        "process samples, not isolated model cost; p95 values are averaged across trials.",
        "",
        "![Budget and latency over time](real-dynamics.png)",
        "",
        "![Traffic accounting over time](real-traffic.png)",
        "",
        "![Safety interventions](safety.png)",
        "",
        "Time-series illustrations use the prespecified first seed (81000), not a "
        "best-performing repetition. Arrival cohorts count successful completions, "
        "including completions after the measurement boundary. All variants' raw traces "
        "remain available locally.",
        "",
        "Intervals describe run-to-run uncertainty with a small number of seeds; "
        "they do not include uncertainty from training seeds or simulator design. "
        "Fast rejections are excluded from successful-response latency percentiles. "
        "Tail values summarize per-run percentiles (means across repetitions), "
        "not a percentile computed from pooling all runs. "
        "Any model fallback or excessive missed launches makes a run ineligible for "
        "a learned-policy performance claim.",
        "",
        "### Paired goodput differences against AIMD",
        "",
        "| Controller | Matched seeds | Difference RPS (95% t interval) |",
        "|---|---:|---:|",
    ]
    aimd = {r["seed"]: r for r in summary_groups.get("adaptive", [])}
    for name, group in summary_groups.items():
        if name == "adaptive":
            continue
        diffs = [
            r["goodput_rps"] - aimd[r["seed"]]["goodput_rps"] for r in group if r["seed"] in aimd
        ]
        if diffs:
            mean, error = interval(diffs)
            text = f"{mean:+.2f} ± {error:.2f}" if error is not None else f"{mean:+.2f} (single)"
            lines.append(f"| {LABELS[name]} | {len(diffs)} | {text} |")
    lines += [
        "",
        "## Held-out simulation",
        "",
        "![Simulation](simulator.png)",
        "",
        f"{len(sim)} episodes: 12 variants, 10 workload families, five matched seeds "
        "(40000–40004), each 180 seconds with 30 seconds warm-up. Static budgets use "
        "the real calibration's 24/36 RPS. Backend-aware PPO is separately identified. "
        "PPO + prediction without safety is a simulator-only ablation; the live hybrid "
        "also has safety. The falling-demand and combined unseen profiles were not "
        "training families. Inference is deterministic in all policy comparisons.",
        "",
        "Highest observed mean by workload (not a significance claim):",
        "",
        "| Family | Controller | Goodput RPS |",
        "|---|---|---:|",
        *winners,
        "",
        "### Stronger conventional comparator (supplemental)",
        "",
        "After inspecting initial results, a separate 16-setting conventional search "
        "used only the six validation seeds and PPO's independent selection score. "
        "It selected a 2s window, EWMA alpha 0.5, one healthy evaluation, +5 RPS steps, "
        "and ten observations. The original baseline is unchanged. This is an exploratory "
        "supplement, not a retrospectively substituted baseline.",
        "",
        f"The frozen tuning averaged {np.mean([r['goodput_rps'] for r in tuned]):.2f} "
        "goodput RPS across the same 50 held-out simulator scenarios, versus 21.93 "
        "for original AIMD and 31.18 for gateway-only PPO. Faster conventional recovery "
        "closes much of the original gap; ML should be compared against such tuning too.",
        "",
        f"Supplemental HTTP status: {len(supplemental)}/3 valid trials. "
        "These run after the main randomized study, with matched seeds and identical "
        "capacity protocol, so host/time ordering remains a possible confound.",
        "",
        supplemental_result,
        "",
        "| Tuned AIMD supplemental HTTP | Goodput RPS | Backend errors % offered | "
        "Successful p99 ms | Offered / launched | Missed launches % | Admission unknown |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| Seed {json.loads((Path(r['run']) / 'experiment.json').read_text())['seed']} | "
            f"{r['goodput_rps']:.2f} | {r['backend_error_fraction'] * 100:.2f} | "
            f"{r['latency_p99'] * 1000:.1f} | {r['offered']} / {r['launched']} | "
            f"{r['missed_launch_fraction'] * 100:.3f} | {r['admission_unknown']} |"
            for r in supplemental
        ],
        "Supplemental trials retain any missed launches under the same preset ≤1% "
        "validity rule. Goodput uses offered measurement duration; missed requests "
        "do not earn useful-work credit and are not silently removed or retried.",
        "One supplemental client-side transport failure has unknown admission. "
        "It earns no goodput; the recorded trace does not identify its root cause. "
        "There were no such failures in the main study.",
        "",
        "## Simulator fidelity",
        "",
        "![Fidelity](fidelity.png)",
        "",
        "![Dynamic fidelity](fidelity-dynamic.png)",
        "",
        "The unthrottled HTTP sweep used 5 seconds warm-up and 15 seconds measurement "
        "at 10, 20, 30, 40, and 60 RPS. The simulator uses nominal service and 5ms "
        "transport overhead without fitting final-test outcomes. Throughput agrees "
        "within about 2% on this sweep; simulated p95 is roughly 11–34ms lower. "
        "Queue saturation and recovery trends agree qualitatively. This supports "
        "coarse dynamics, not exact tail fidelity or transfer to arbitrary real APIs.",
        "",
        "## Training and resource choice",
        "",
        "![Training](training.png)",
        "",
        "![Reward components](reward-components.png)",
        "",
        "![Hardware](hardware.png)",
        "",
        "Each selected PPO variant trained for 131,072 transitions with four CPU "
        "subprocess environments, 256 rollout steps per environment, batch size 256, "
        "six epochs, learning rate 0.0003, gamma 0.98, GAE lambda 0.95, and entropy "
        "coefficient 0.01. Actor and value networks each use two 64-unit Tanh layers. "
        "Checkpoint selection uses six fixed validation scenarios, not held-out tests. "
        "Selected deployment actors are exported from those checkpoints; they are not heuristics.",
        "The selected checkpoints are 131,072 transitions for gateway-only PPO, "
        "98,304 for backend-aware PPO, and 32,768 for prediction-aware PPO. "
        "Later training did not consistently improve validation performance.",
        "",
        "### Environment, observations, actions, and reward",
        "",
        "A request-level event heap models arrivals, atomic global/client token budgets, "
        "finite workers/queue, overflow, delayed completions, and timeout work that "
        "continues on the backend. Traces are generated before policy decisions. "
        "Randomized service/capacity/load episodes use Gymnasium time-limit truncation "
        "and pass SB3's environment checker. There are no real-time training sleeps.",
        "",
        "Six actions multiply the budget by 0.8, 0.9, 0.95, 1, 1.05, or 1.1. "
        "Sixteen shared normalized gateway features contain current budget, "
        "offered/admitted/completed/goodput RPS, latency EWMA/p90, error/timeout/rejection "
        "fractions, in-flight utilization, previous action, demand/latency trends, sample "
        "volume and an availability mask. Backend-aware PPO adds queue/busy utilization "
        "and a telemetry mask; prediction-aware PPO adds forecast risk. Fixed physical "
        "scaling, order, and clipping are identical in training and inference. "
        "No future capacity or worker-count oracle enters the gateway-only actor.",
        "",
        "For offered O, completed C, timely successes G, late successes L, errors E, "
        "timeouts T, relative budget change u, and D=max(1,O,C), the reward is:",
        "",
        "```text",
        "G/D - 0.70 L/D - 2.00 (E+T)/D - 0.03 rejected/max(1,O)",
        "    - 0.02 abs(u) - 0.02 (1-Jain)",
        "```",
        "",
        "Fairness applies only with equal demands and at least six successes. "
        "Useful completions earn reward; admission alone does not. Reject-all loses "
        "useful-work reward and pays rejection penalties. Tests cover no demand, useful "
        "service, reject-all and overload; this does not prove absence of reward hacking. "
        "Reported HTTP goodput uses end-user latency; the live feature uses upstream "
        "timing as an available estimate.",
        "",
        "The 2,048-transition hardware probe used 128 rollout steps and three epochs. "
        "Four CPU subprocesses measured about 2,539 transitions/s versus 1,450 with CUDA. "
        "CUDA was available on the RTX 4070 SUPER, but CPU was faster for this MLP "
        "probe. These are single throughput probes, not statistically repeated speed claims. "
        "Live actors run with NumPy, verified against PPO actions, without loading "
        "PyTorch or running inference on individual requests.",
        "",
        "## Optional predictor",
        "",
        "![Prediction](prediction.png)",
        "",
        f"Scenario-held-out ROC AUC: {predictor_metrics['roc_auc']:.4f}; "
        f"average precision: {predictor_metrics['average_precision']:.4f}; "
        f"Brier: {predictor_metrics['brier']:.4f}, versus "
        f"{predictor_metrics['constant_brier']:.4f} for constant prevalence and "
        f"{predictor_metrics.get('persistence_brier', float('nan')):.4f} for "
        "current-health persistence (p90 above target or error fraction above 2%). "
        "Persistence is a stronger comparator than constant prevalence.",
        "",
        "A calibrated histogram gradient-boosted classifier predicts whether any of "
        "the next three seconds violates latency/error thresholds under the sampled "
        "behavior policy. Fitting, calibration, and testing use whole disjoint scenario "
        "seed groups. Held-out AUC/Brier alone do not prove proactive prediction of "
        "unannounced capacity failures, or a causal outcome for a proposed budget. "
        "Simulator ablations show little additional goodput from prediction or backend "
        "instrumentation over the simpler gateway-only PPO. They remain optional.",
        "",
        "## Engineering, limitations, and reproduction",
        "",
        "A post-training simulator edge audit found raw PPO raises its budget to "
        "120 RPS even with no demand, without earning useful work. Safety holds "
        "the initial budget when there are no samples. With only 5 RPS demand, both "
        "PPO variants can raise the unused budget to the ceiling: sparse but steady "
        "successful samples satisfy the current guard. This can increase exposure "
        "to a later burst and is an unresolved demand-awareness weakness. When "
        "capacity is just 2 RPS, every controller's 10 RPS floor is too high and "
        "timely goodput falls to zero. No claim of universal safe operation is made.",
        "",
        "See [RL implementation and commands](../../docs/rl.md). The original Redis "
        "algorithms and EWMA/AIMD policy remain available. The safety layer guards "
        "bounds, jumps, timeouts, unhealthy increases, and sparse samples; explicit "
        "inference/load failures fall back to AIMD and are logged. Model files and "
        "normalization are validated at startup.",
        "",
        "Limitations include one trained seed per selected architecture, a single "
        "controlled backend and host, simplified uniform/lognormal service distributions, "
        "fixed equal client quotas, a minimum budget that can itself exceed very small "
        "backend capacity, simulation timing approximation, and limited real workload "
        "coverage. Three workload repetitions do not establish generalization across "
        "hardware or retraining variability. Docker configuration and packaging can "
        "be checked locally, but a complete local Docker/Grafana run is unavailable "
        "while the Docker engine is stopped. CI's previous Linux typing issue was "
        "identified and repaired; a new remote CI run has not been claimed.",
        "",
        "Full request CSVs, training logs, checkpoints, simulator episode records and "
        "source/model snapshots are retained under ignored `results/ml-runs/` and "
        "`models/`. Reviewed summaries and model provenance accompany this report. "
        "The owner's extended guide remains local and Git-ignored.",
        "",
    ]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    save_json(output / "real-summaries.json", real)
    save_json(output / "simulator-summaries.json", sim)
    if tuned:
        save_json(output / "tuned-aimd-simulator.json", tuned)
        shutil.copy2("models/aimd-tuned.json", output / "aimd-selection.json")
    if edges:
        save_json(output / "edge-audit.json", edges)
    if supplemental:
        save_json(output / "tuned-aimd-real.json", supplemental)
    save_json(
        output / "status.json",
        {
            "complete": complete,
            "real_completed": len(real),
            "real_expected": expected,
            "real_valid": len(valid),
            "simulator_episodes": len(sim),
            "supplemental_valid": len(supplemental),
        },
    )
    for source, target in [
        (Path("results/ml-runs/hardware.json"), "hardware.json"),
        (fidelity_root / "steady-fidelity.json", "fidelity.json"),
        (Path("models/overload-predictor/metadata.json"), "predictor.json"),
    ]:
        shutil.copy2(source, output / target)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/rl-upgrade"))
    args = parser.parse_args()
    print(build_upgrade_report(args.real, args.sim, args.fidelity, args.output))
