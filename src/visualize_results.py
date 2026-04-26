import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import matplotlib.ticker as mticker
import os

REPORT_PATH = "data/outputs/final_comparison_report.csv"
OUTPUT_DIR  = "results/"

def plot_scorecard(report_path: str = REPORT_PATH) -> None:
    df = pd.read_csv(report_path).set_index("Model")

    quality_cols = [c for c in df.columns if c != "avg_latency_seconds"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model comparison scorecard", fontsize=14, fontweight="medium", y=1.02)

    # ── Quality metrics (grouped bar) ─────────────────────────────────────
    df[quality_cols].plot(kind="bar", ax=axes[0], color=["#70B3EE", "#F7AAD3"],
                          edgecolor="none", width=0.6)
    axes[0].set_title("Quality metrics", fontsize=15, fontweight="medium")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    axes[0].legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", fontsize=8, padding=3)
    # ── Latency (horizontal bar) ──────────────────────────────────────────
    df["avg_latency_seconds"].plot(kind="barh", ax=axes[1],
                                   color=["#70B3EE", "#F7AAD3"], edgecolor="none")
    axes[1].set_title("Average Latency (s)", fontsize=12, fontweight="medium")
    axes[1].set_xlabel("Seconds")
    axes[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0fs"))
    axes[1].spines[["top", "right"]].set_visible(False)

    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", fontsize=8, padding=3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "scorecard.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {out_path}")

if __name__ == "__main__":
    plot_scorecard()