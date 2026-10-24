"""Render a reproducible architecture figure without a browser or external service."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.5), layout="constrained")
    fig.patch.set_facecolor("#f7f9fc")
    ax.set(xlim=(0, 15), ylim=(0, 8.5))
    ax.axis("off")
    ax.text(0.4, 8, "INTELLIGENT API RATE LIMITING", fontsize=21, weight="bold", color="#142b46")
    ax.text(
        0.4,
        7.55,
        "Request admission · feedback control · reproducible evidence",
        fontsize=12,
        color="#52667d",
    )
    nodes = {
        "load": (1.8, 5.8, "Traffic generator", "Seeded open arrivals"),
        "gateway": (6.2, 5.8, "FastAPI gateway", "Identity · admission · proxy"),
        "backend": (12.1, 5.8, "Backend API", "Bounded workers + queue"),
        "controller": (6.2, 3.5, "Adaptive controller", "EWMA + AIMD · every second"),
        "redis": (10.1, 3.5, "Redis", "Atomic budgets + client state"),
        "raw": (1.8, 3.5, "Run artifacts", "CSV · manifests · decisions"),
        "report": (1.8, 1.2, "Comparison report", "Goodput · tails · fairness"),
        "prom": (9.3, 1.2, "Prometheus", "Request + backend metrics"),
        "grafana": (13.1, 1.2, "Grafana", "Live dashboard"),
    }
    width, height = 2.8, 1.05
    for key, (x, y, title, subtitle) in nodes.items():
        fill = "#fff0df" if key == "controller" else "white"
        edge = "#bc7934" if key == "controller" else "#b8c8da"
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.12, title, ha="center", fontsize=12, weight="bold", color="#142b46")
        ax.text(x, y - 0.22, subtitle, ha="center", fontsize=9, color="#52667d")

    def arrow(start, end, label="", label_offset=(0, 0.15), color="#607890"):
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=13, color=color, linewidth=1.5
            )
        )
        if label:
            ax.text(
                (start[0] + end[0]) / 2 + label_offset[0],
                (start[1] + end[1]) / 2 + label_offset[1],
                label,
                ha="center",
                fontsize=9,
                color=color,
                bbox={"facecolor": "#f7f9fc", "edgecolor": "none", "pad": 1},
            )

    arrow((3.3, 5.8), (4.7, 5.8), "HTTP")
    arrow((7.7, 5.8), (10.6, 5.8), "Forward / response")
    arrow((6.2, 5.15), (6.2, 4.12), "Observations", (0.65, 0))
    arrow((7.7, 3.5), (8.6, 3.5), "Budget")
    arrow((7.7, 5.3), (9.5, 4.12), "Admission")
    arrow((1.8, 5.15), (1.8, 4.12), "Raw records", (0.65, 0))
    arrow((4.7, 3.5), (3.3, 3.5), "Decisions")
    arrow((1.8, 2.85), (1.8, 1.82), "Analyze", (0.55, 0))
    # Metrics use the clear corridors between nodes rather than crossing Redis.
    ax.plot([12.1, 12.1, 10.2], [5.15, 2.2, 2.2], color="#607890", linestyle="--")
    arrow((10.2, 2.2), (10.2, 1.82))
    ax.text(12.25, 2.8, "Scrape", fontsize=9, color="#607890")
    ax.plot([7.5, 8.0, 8.0, 9.0], [5.15, 4.7, 2.2, 2.2], color="#607890", linestyle="--")
    arrow((9.0, 2.2), (9.0, 1.82))
    ax.text(7.85, 2.65, "Scrape", ha="right", fontsize=9, color="#607890")
    arrow((10.8, 1.2), (11.6, 1.2))
    ax.text(
        4.7,
        0.25,
        "Single gateway process · single upstream · registered clients · local deployment",
        fontsize=10,
        color="#52667d",
    )
    output = Path("docs")
    output.mkdir(exist_ok=True)
    fig.savefig(output / "architecture.png", dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(output / "architecture.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
