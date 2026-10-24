"""Rebuildable reports from raw request records; never average per-run percentiles."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def summarize(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    with (run / "requests.csv").open(encoding="utf-8", newline="") as file:
        rows = [r for r in csv.DictReader(file) if float(r["offset"]) >= -1e-6]
    duration = sum(p["duration"] for p in manifest["scenario"]["phases"])
    target = manifest["settings"]["controller"]["latency_target"]
    successes = [r for r in rows if r["outcome"] == "success"]
    good = [r for r in successes if float(r["latency"]) <= target]
    outcomes = Counter(r["outcome"] for r in rows)
    per_client = {
        name: sum(r["client"] == name for r in successes) / duration for name in manifest["clients"]
    }
    values = list(per_client.values())
    fairness = (
        sum(values) ** 2 / (len(values) * sum(x * x for x in values))
        if any(values) and manifest["scenario"]["fairness_comparable"]
        else None
    )
    latencies = [float(r["latency"]) for r in successes]
    missed = outcomes["missed_launch"]
    unknown = outcomes["client_timeout"] + outcomes["transport_error"]
    summary: dict[str, Any] = {
        "run": str(run),
        "policy": manifest["settings"]["policy"],
        "rate": manifest["settings"]["rate"],
        "scenario": manifest["scenario_name"],
        "complete": manifest["complete"],
        "duration": duration,
        "offered": len(rows),
        "launched": len(rows) - missed,
        "admitted": sum(r["admitted"] == "True" for r in rows),
        "admission_unknown": unknown,
        "completed": sum(bool(r["status"]) for r in rows),
        "outcomes": dict(outcomes),
        "success_rps": len(successes) / duration,
        "goodput_rps": len(good) / duration,
        "latency_p50": quantile(latencies, 0.5),
        "latency_p95": quantile(latencies, 0.95),
        "latency_p99": quantile(latencies, 0.99),
        "rejection_p95": quantile(
            [float(r["latency"]) for r in rows if r["outcome"] == "rejected"], 0.95
        ),
        "admission_overhead_p95_ms": quantile(
            [float(r["admission_ms"]) for r in rows if r["admission_ms"]], 0.95
        ),
        "launch_lag_p99": quantile([float(r["launch_lag"]) for r in rows], 0.99),
        "per_client_success_rps": per_client,
        "jain_fairness": fairness,
        "valid_generator": missed / len(rows) <= 0.01 if rows else False,
        "recovery_seconds": None,
    }
    events_path = run / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    # Recovery: three consecutive 5s windows with >=80% of pre-degradation goodput
    # and successful-request p95 <= target. Only score actual capacity restorations.
    if len(events) >= 3 and events[-1]["workers"] > events[-2]["workers"]:
        degraded, recovered = events[-2]["offset"], events[-1]["offset"]
        before = [r for r in good if max(0, degraded - 30) <= float(r["offset"]) < degraded]
        baseline = len(before) / min(30, max(degraded, 0.001))
        consecutive = 0
        for start in np.arange(recovered, duration - 5, 5):
            window = [r for r in successes if start <= float(r["offset"]) < start + 5]
            p95 = quantile([float(r["latency"]) for r in window], 0.95)
            gp = sum(float(r["latency"]) <= target for r in window) / 5
            consecutive = (
                consecutive + 1
                if baseline > 0 and gp >= 0.8 * baseline and (p95 is not None and p95 <= target)
                else 0
            )
            if consecutive == 3:
                summary["recovery_seconds"] = float(start + 5 - recovered)
                break
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_run(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    with (run / "requests.csv").open(newline="") as file:
        rows = [r for r in csv.DictReader(file) if float(r["offset"]) >= 0]
    duration = sum(p["duration"] for p in manifest["scenario"]["phases"])
    target = manifest["settings"]["controller"]["latency_target"]
    width = max(1.0, duration / 100)
    edges = np.arange(0, duration + width, width)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for label, subset in (
        ("offered", rows),
        (
            "goodput",
            [r for r in rows if r["outcome"] == "success" and float(r["latency"]) <= target],
        ),
        ("rejected", [r for r in rows if r["outcome"] == "rejected"]),
        (
            "errors",
            [r for r in rows if r["outcome"] not in {"success", "rejected", "missed_launch"}],
        ),
    ):
        counts, _ = np.histogram([float(r["offset"]) for r in subset], bins=edges)
        axes[0].step(edges[:-1], counts / width, where="post", label=label)
    axes[0].set_ylabel("Requests / second")
    axes[0].legend(ncol=4)
    p95 = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        values = [
            float(r["latency"]) * 1000
            for r in rows
            if r["outcome"] == "success" and left <= float(r["offset"]) < right
        ]
        p95.append(quantile(values, 0.95) or float("nan"))
    axes[1].plot(edges[:-1], p95, label="successful p95")
    axes[1].axhline(target * 1000, color="red", linestyle="--", label="target")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].legend()
    decision_path = run / "decisions.jsonl"
    if decision_path.exists():
        decisions = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
        origin = manifest.get("measured_start_wall", manifest["started_at"])
        applied = [d for d in decisions if d["applied"]]
        axes[2].step(
            [d["wall_time"] - origin for d in applied],
            [d["rate"] for d in applied],
            where="post",
            label="applied budget",
        )
    else:
        if manifest["settings"]["policy"] == "none":
            axes[2].text(
                0.5,
                0.5,
                "Rate limiter disabled",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
            )
        else:
            axes[2].axhline(manifest["settings"]["rate"], label="configured budget")
    axes[2].set_ylabel("Budget (RPS)")
    axes[2].set_xlabel("Measured workload time (seconds)")
    axes[2].set_xlim(0, duration)
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"{manifest['scenario_name']} / {manifest['settings']['policy']}")
    fig.savefig(run / "timeseries.png", dpi=150)
    plt.close(fig)


def build_report(root: Path) -> Path:
    manifests = sorted(root.rglob("manifest.json"))
    runs = [p.parent for p in manifests]
    summaries = [summarize(run) for run in runs]
    if not summaries:
        raise ValueError(f"No run manifests found under {root}")
    for run in runs:
        plot_run(run)
    groups: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        policy_label = (
            "no limiter"
            if summary["policy"] == "none"
            else f"{summary['policy']} @ {summary['rate']:g}"
        )
        key = f"{summary['scenario']} / {policy_label}"
        groups.setdefault(key, []).append(summary)
    scenarios = list(dict.fromkeys(s["scenario"] for s in summaries))
    fig, grid = plt.subplots(
        len(scenarios), 2, figsize=(13, 4.3 * len(scenarios)), squeeze=False, layout="constrained"
    )
    for axes, scenario in zip(grid, scenarios, strict=True):
        labels = [label for label, group in groups.items() if group[0]["scenario"] == scenario]
        display = [
            label.split(" / ", 1)[1].replace("_", " ").replace(" @ ", "\n") for label in labels
        ]
        colors = [
            "#bd7837" if groups[label][0]["policy"] == "adaptive" else "#356b86" for label in labels
        ]
        for ax, metric, title in zip(
            axes,
            ("goodput_rps", "latency_p99"),
            ("Goodput (RPS)", "Successful p99 (seconds)"),
            strict=True,
        ):
            values = [
                [
                    r[metric]
                    for r in groups[label]
                    if r[metric] is not None and r["complete"] and r["valid_generator"]
                ]
                for label in labels
            ]
            means = [float(np.mean(v)) if v else float("nan") for v in values]
            positions = np.arange(len(labels))
            ax.bar(positions, means, color=colors)
            for position, observations in enumerate(values):
                if len(observations) > 1:
                    ax.errorbar(
                        position,
                        means[position],
                        yerr=np.std(observations, ddof=1),
                        fmt="none",
                        color="black",
                        capsize=4,
                    )
            ax.set_xticks(positions, display, fontsize=9)
            if not any(values):
                ax.text(
                    0.5,
                    0.6,
                    "No valid complete runs\nSee missed-launch counts in the report",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_ylim(0, 1)
            ax.set_title(f"{scenario} · {title}")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Complete, generator-valid runs · means; sample SD bars only when n ≥ 2")
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Benchmark results",
        "",
        "Measured outcomes, not a claim that adaptation always wins. Error bars show sample "
        "standard deviation across complete runs with at most 1% missed launches. "
        "Percentiles are computed within each run; the chart compares those run statistics.",
        "A single trial has no variability estimate; it is not evidence of repeatability.",
        "",
        "![Comparison](comparison.png)",
        "",
        "| Scenario / policy @ initial RPS | Runs | Goodput mean ± SD | Successful p99 range (s) |",
        "|---|---:|---:|---:|",
    ]
    for label, group in groups.items():
        valid = [r for r in group if r["complete"] and r["valid_generator"]]
        gp = [r["goodput_rps"] for r in valid]
        tails = [r["latency_p99"] for r in valid if r["latency_p99"] is not None]
        mean = (
            f"{np.mean(gp):.2f} ± {np.std(gp, ddof=1):.2f}"
            if len(gp) > 1
            else f"{gp[0]:.2f} (single trial)"
            if gp
            else "N/A"
        )
        tail = f"{min(tails):.3f}–{max(tails):.3f}" if tails else "N/A"
        lines.append(f"| {label} | {len(valid)}/{len(group)} | {mean} | {tail} |")
    lines += [
        "",
        "## Individual runs",
        "",
        "Admission counts exclude transport failures with unknown admission. "
        "Offered = launched + missed; launched requests are partitioned by outcome. "
        "Requests are assigned to measurement by scheduled arrival, including completions "
        "during drain. Recovery requires three healthy 5-second windows. Fairness is only "
        "scored for scenarios marked as comparable equal demand.",
        "",
    ]
    for run, summary in zip(runs, summaries, strict=True):
        relative = run.relative_to(root).as_posix()
        lines.append(
            f"- [{relative or '.'}]({relative}/summary.json): "
            f"offered={summary['offered']}, launched={summary['launched']}, "
            f"outcomes={summary['outcomes']}, fairness={summary['jain_fairness']}, "
            f"recovery={summary['recovery_seconds']}, "
            f"generator_valid={summary['valid_generator']}. "
            f"[Time series]({relative}/timeseries.png)"
        )
    path = root / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return path
