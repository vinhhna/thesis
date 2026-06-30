from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "report_figures"


def save_figure(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, width, height, title, body, face, edge):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#17202a",
    )
    ax.text(
        x + width / 2,
        y + height * 0.36,
        body,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#2f3b45",
        linespacing=1.25,
    )
    return box


def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color="#4a5568",
        shrinkA=7,
        shrinkB=7,
    )
    ax.add_patch(arrow)


def build_method_pipeline():
    fig, ax = plt.subplots(figsize=(12.4, 4.1))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.text(
        0.25,
        3.82,
        "Redundancy-aware visual-token selection used by Ours",
        fontsize=15,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="center",
    )
    ax.text(
        0.25,
        3.54,
        "The SparseVLM relevance signal is preserved, then selected tokens are diversified before inference continues.",
        fontsize=10,
        color="#4b5563",
        ha="left",
        va="center",
    )

    colors = {
        "blue_face": "#e8f1ff",
        "blue_edge": "#2f6fbb",
        "green_face": "#e9f7ef",
        "green_edge": "#2e7d32",
        "gold_face": "#fff4d8",
        "gold_edge": "#b7791f",
        "red_face": "#fdecec",
        "red_edge": "#b23b3b",
        "gray_face": "#f3f4f6",
        "gray_edge": "#5b6472",
    }

    x_positions = [0.25, 2.65, 5.05, 7.45, 9.85]
    y = 1.65
    w = 1.9
    h = 1.25
    boxes = [
        draw_box(
            ax,
            (x_positions[0], y),
            w,
            h,
            "Relevance scores",
            "Text-guided\nSparseVLM token\nimportance",
            colors["blue_face"],
            colors["blue_edge"],
        ),
        draw_box(
            ax,
            (x_positions[1], y),
            w,
            h,
            "Candidate pool",
            "Top $\\alpha k_\\ell$\nhigh-relevance\nvisual tokens",
            colors["green_face"],
            colors["green_edge"],
        ),
        draw_box(
            ax,
            (x_positions[2], y),
            w,
            h,
            "Redundancy",
            "Cosine similarity\nto the already\nselected set",
            colors["gold_face"],
            colors["gold_edge"],
        ),
        draw_box(
            ax,
            (x_positions[3], y),
            w,
            h,
            "MMR-style score",
            "$\\lambda$ relevance\nminus redundancy\npenalty",
            colors["red_face"],
            colors["red_edge"],
        ),
        draw_box(
            ax,
            (x_positions[4], y),
            w,
            h,
            "Retained tokens",
            "Fixed sparse budget\npassed to later\nVLM inference",
            colors["gray_face"],
            colors["gray_edge"],
        ),
    ]

    for left, right in zip(boxes, boxes[1:]):
        draw_arrow(
            ax,
            (left.get_x() + left.get_width(), left.get_y() + left.get_height() / 2),
            (right.get_x(), right.get_y() + right.get_height() / 2),
        )

    # Small feedback arc to show greedy updates after each token is selected.
    arc = FancyArrowPatch(
        (8.35, 1.45),
        (5.95, 1.45),
        connectionstyle="arc3,rad=-0.35",
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        linestyle="--",
        color="#6b7280",
    )
    ax.add_patch(arc)
    ax.text(
        7.15,
        0.92,
        "Greedy update: recompute redundancy after adding each selected token",
        fontsize=9.2,
        color="#4b5563",
        ha="center",
        va="center",
    )

    ax.text(
        0.25,
        0.52,
        "Applied at SparseVLM pruning layers while keeping the original model and token-recycling pipeline unchanged.",
        fontsize=9.6,
        color="#374151",
        ha="left",
        va="center",
    )
    ax.text(
        0.25,
        0.27,
        r"$\alpha$: candidate-pool factor; $\lambda$: relevance weight; $k_\ell$: layer-wise sparse token budget.",
        fontsize=8.8,
        color="#6b7280",
        ha="left",
        va="center",
    )

    save_figure(fig, "method_pipeline_ours_selection")


def variant_label(row):
    if row["method"] == "SparseVLM-Original":
        return "SparseVLM\nOriginal"
    if row["method"] == "Threshold-Fixed-k":
        return "Threshold\nFixed"
    return (
        f"Ours\n$\\alpha={int(row['candidate_pool_factor'])}$\n"
        f"$\\lambda={row['lambda_relevance']:.1f}$"
    )


def build_ablation_tradeoff():
    data_path = ROOT / "outputs" / "stage8" / "stage8_ablation_results.csv"
    df = pd.read_csv(data_path)
    df = df[df["dataset"].isin(["gqa", "pope"])].copy()
    df["candidate_pool_factor"] = pd.to_numeric(df["candidate_pool_factor"], errors="coerce")
    df["lambda_relevance"] = pd.to_numeric(df["lambda_relevance"], errors="coerce")
    df["accuracy_pct"] = pd.to_numeric(df["accuracy"], errors="coerce") * 100.0
    df["mean_pairwise_similarity"] = pd.to_numeric(
        df["mean_pairwise_similarity"], errors="coerce"
    )
    df["label"] = df.apply(variant_label, axis=1)

    order = [
        "SparseVLM\nOriginal",
        "Ours\n$\\alpha=2$\n$\\lambda=0.8$",
        "Ours\n$\\alpha=2$\n$\\lambda=0.5$",
        "Ours\n$\\alpha=2$\n$\\lambda=0.7$",
        "Ours\n$\\alpha=3$\n$\\lambda=0.5$",
        "Ours\n$\\alpha=3$\n$\\lambda=0.7$",
        "Threshold\nFixed",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=False)
    fig.patch.set_facecolor("white")

    bar_colors = {
        "SparseVLM": "#94a3b8",
        "Ours": "#3b82f6",
        "Threshold": "#f59e0b",
    }

    for ax, dataset, title in zip(axes, ["gqa", "pope"], ["GQA subset", "POPE subset"]):
        sub = df[df["dataset"] == dataset].copy()
        sub["label"] = pd.Categorical(sub["label"], categories=order, ordered=True)
        sub = sub.sort_values("label")
        labels = sub["label"].astype(str).tolist()
        x = np.arange(len(sub))

        colors = []
        for label in labels:
            if label.startswith("SparseVLM"):
                colors.append(bar_colors["SparseVLM"])
            elif label.startswith("Threshold"):
                colors.append(bar_colors["Threshold"])
            else:
                colors.append(bar_colors["Ours"])

        bars = ax.bar(x, sub["accuracy_pct"], color=colors, edgecolor="#263238", linewidth=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", color="#d9dee7", linestyle="-", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

        low = np.floor((sub["accuracy_pct"].min() - 1.0) / 2.0) * 2.0
        high = np.ceil((sub["accuracy_pct"].max() + 1.0) / 2.0) * 2.0
        ax.set_ylim(low, high)

        for rect, value in zip(bars, sub["accuracy_pct"]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.15,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#263238",
            )

        ax2 = ax.twinx()
        ax2.plot(
            x,
            sub["mean_pairwise_similarity"],
            color="#111827",
            marker="o",
            markersize=5,
            linewidth=1.5,
            label="Mean pairwise similarity",
        )
        ax2.set_ylabel("Mean pairwise similarity", fontsize=10)
        ax2.set_ylim(0.18, 0.31)
        ax2.tick_params(axis="y", labelsize=8.5)

        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#cbd5e1")
        ax2.spines["top"].set_color("#cbd5e1")
        ax2.spines["right"].set_color("#cbd5e1")

    handles = [
        plt.Line2D([0], [0], color=bar_colors["SparseVLM"], lw=8, label="SparseVLM-Original"),
        plt.Line2D([0], [0], color=bar_colors["Ours"], lw=8, label="Ours variants"),
        plt.Line2D([0], [0], color=bar_colors["Threshold"], lw=8, label="Threshold-Fixed"),
        plt.Line2D(
            [0],
            [0],
            color="#111827",
            marker="o",
            lw=1.5,
            label="Mean pairwise similarity",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False)
    fig.suptitle(
        "Ablation results: answer accuracy and selected-token similarity",
        fontsize=15,
        fontweight="bold",
        y=1.13,
    )
    fig.text(
        0.5,
        -0.02,
        "Subset size: 500 examples per dataset. These ablation results are auxiliary and do not replace the main benchmark table.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    save_figure(fig, "stage8_ablation_accuracy_similarity")


def main():
    build_method_pipeline()
    build_ablation_tradeoff()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
