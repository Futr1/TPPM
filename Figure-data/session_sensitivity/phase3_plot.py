#!/usr/bin/env python3
"""Phase 3: Generate 1×3 panel figure for session count sensitivity.

Reads aggregate_results.json from Phase 2 and produces:
  - session_sensitivity.pdf (vector)
  - session_sensitivity.png (raster)

Usage:
    python3 phase3_plot.py
    python3 phase3_plot.py --input path/to/aggregate_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Figure-data/session_sensitivity")
DEFAULT_INPUT = ROOT / "eval_results" / "aggregate_results.json"
FIGURE_DIR = ROOT / "figures"

# ===== Style =====
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Colors
COLOR_OVERALL = "#2563EB"     # blue
COLOR_ANSWERABLE = "#DC2626"  # red
COLOR_UPR = "#059669"         # green


def load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_data(
    results: list[dict],
    metric: str,
) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    """Extract metric data across N values.

    Returns:
        (overall_means, overall_ses, answerable_means, answerable_ses, n_values)
    """
    overall_means = []
    overall_ses = []
    answerable_means = []
    answerable_ses = []
    n_values = []

    for entry in results:
        n = entry["n"]
        n_values.append(n)

        ovr = entry.get("overall", {}).get(metric, {})
        ans = entry.get("answerable", {}).get(metric, {})

        overall_means.append(ovr.get("mean", 0.0))
        overall_ses.append(ovr.get("se", 0.0))
        answerable_means.append(ans.get("mean", 0.0))
        answerable_ses.append(ans.get("se", 0.0))

    return overall_means, overall_ses, answerable_means, answerable_ses, n_values


def plot_panel(
    ax: plt.Axes,
    n_values: list[int],
    overall_means: list[float],
    overall_ses: list[float],
    answerable_means: list[float],
    answerable_ses: list[float],
    ylabel: str,
    title: str,
    lower_is_better: bool = False,
) -> None:
    """Plot one panel with dual lines (Overall + Answerable)."""
    x = np.arange(len(n_values))

    # Overall F1 — solid line
    ax.errorbar(x, overall_means, yerr=overall_ses,
                fmt="o-", color=COLOR_OVERALL, linewidth=1.8,
                markersize=5, capsize=3, label="Overall F1",
                zorder=3)
    ax.fill_between(x,
                    np.array(overall_means) - np.array(overall_ses),
                    np.array(overall_means) + np.array(overall_ses),
                    color=COLOR_OVERALL, alpha=0.12, zorder=2)

    # Answerable F1 — dashed line
    ax.errorbar(x, answerable_means, yerr=answerable_ses,
                fmt="s--", color=COLOR_ANSWERABLE, linewidth=1.5,
                markersize=4, capsize=3, label="Answerable F1",
                zorder=3)
    ax.fill_between(x,
                    np.array(answerable_means) - np.array(answerable_ses),
                    np.array(answerable_means) + np.array(answerable_ses),
                    color=COLOR_ANSWERABLE, alpha=0.10, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("Session Count (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Annotate direction
    if lower_is_better:
        ax.annotate("↓ better", xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color="gray")
    else:
        ax.annotate("↑ better", xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color="gray")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3: Generate session sensitivity figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()

    results = load_results(args.input)
    if not results:
        print("[ERROR] No results found.")
        return 1

    print(f"[INFO] Loaded {len(results)} N-value results")

    # Create 1×3 figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    metrics = [
        ("overall", "F1 Score", "(a) QA Overall F1", False),
        ("temporal", "F1 Score", "(b) QA Temporal F1", False),
        # For UPR, we'd need separate data — for now, reuse overall as placeholder
        # The actual UPR data would come from profile evaluation
    ]

    # Panel (a): Overall F1
    ovr_m, ovr_se, ans_m, ans_se, n_vals = extract_data(results, "overall")
    plot_panel(axes[0], n_vals, ovr_m, ovr_se, ans_m, ans_se,
               ylabel="F1 Score", title="(a) QA Overall F1")

    # Panel (b): Temporal F1
    ovr_m, ovr_se, ans_m, ans_se, n_vals = extract_data(results, "temporal")
    plot_panel(axes[1], n_vals, ovr_m, ovr_se, ans_m, ans_se,
               ylabel="F1 Score", title="(b) QA Temporal F1")

    # Panel (c): Multi-hop F1 (as proxy — UPR needs separate evaluation)
    ovr_m, ovr_se, ans_m, ans_se, n_vals = extract_data(results, "multi_hop")
    plot_panel(axes[2], n_vals, ovr_m, ovr_se, ans_m, ans_se,
               ylabel="F1 Score", title="(c) QA Multi-hop F1")

    plt.tight_layout(w_pad=2.5)

    # Save
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / "session_sensitivity.pdf"
    png_path = FIGURE_DIR / "session_sensitivity.png"

    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)

    print(f"[DONE] Saved: {pdf_path}")
    print(f"[DONE] Saved: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
