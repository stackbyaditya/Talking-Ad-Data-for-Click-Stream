"""Generate a publication-quality architecture diagram for the full ML pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = MODEL_ROOT / "analysis" / "plots"
OUTPUT_PATH = PLOTS_DIR / "project_architecture_diagram.png"


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#1F2933",
) -> None:
    """Draw a rounded rectangular stage block with title and content."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#102A43",
    )
    ax.text(
        x + 0.02,
        y + height - 0.085,
        body,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#243B53",
        linespacing=1.35,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#334E68") -> None:
    """Draw a directional connector arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def build_matplotlib_diagram(output_path: Path) -> None:
    """Render the project architecture diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.16, 1)
    ax.axis("off")

    fig.suptitle(
        "Click Fraud Detection System Architecture",
        fontsize=24,
        fontweight="bold",
        color="#102A43",
        y=0.98,
    )
    ax.text(
        0.5,
        0.945,
        "Behavioral Clickstream Analytics Pipeline for TalkingData AdTracking Fraud Detection",
        ha="center",
        va="center",
        fontsize=13,
        color="#486581",
    )

    common_w = 0.34
    common_h = 0.11
    center_x = 0.33

    add_box(
        ax,
        center_x,
        0.81,
        common_w,
        common_h,
        "1. Data Source",
        "TalkingData AdTracking Dataset (Kaggle)\n\nRaw features:\nip, app, device, os, channel,\nclick_time, is_attributed",
        "#DCEBFA",
    )
    add_box(
        ax,
        center_x,
        0.67,
        common_w,
        common_h,
        "2. Sessionization",
        "Group click events into user sessions\n\nKeys: (ip, app, device, os, channel)\nBoundary rule: 30-minute inactivity gap\nOutput: session-level records",
        "#E4F7E7",
    )
    add_box(
        ax,
        center_x,
        0.53,
        common_w,
        common_h,
        "3. Synthetic Behavioral Dataset",
        "Generated due to mostly single-click raw sessions\n\nFinal dataset: 9000 sessions\nHuman: 5000\nModerate Bot: 2000\nAdvanced Bot: 2000",
        "#FFF3D6",
    )
    add_box(
        ax,
        center_x,
        0.37,
        common_w,
        0.13,
        "4. Feature Engineering",
        "41 behavioral features across four groups\n\nBehavioral: mouse dynamics, entropy, smoothness\nTemporal: rates, intervals, burst patterns\nNetwork: browser, OS, device, geo, proxy\nHeuristic: bot score, anomaly, success, burstiness",
        "#FDE2E4",
    )
    add_box(
        ax,
        center_x,
        0.21,
        common_w,
        0.11,
        "5. Data Preprocessing",
        "Drop metadata columns\nOneHotEncoder for categorical variables\nRobustScaler for numerical features\nTrain/test split",
        "#EADCF8",
    )

    add_arrow(ax, (0.5, 0.81), (0.5, 0.78))
    add_arrow(ax, (0.5, 0.67), (0.5, 0.64))
    add_arrow(ax, (0.5, 0.53), (0.5, 0.50))
    add_arrow(ax, (0.5, 0.37), (0.5, 0.32))

    ax.text(
        0.5,
        0.18,
        "6. Dual-Stream Modeling Framework",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#102A43",
    )

    add_arrow(ax, (0.5, 0.21), (0.33, 0.145))
    add_arrow(ax, (0.5, 0.21), (0.67, 0.145))

    add_box(
        ax,
        0.08,
        0.06,
        0.34,
        0.11,
        "6A. Tabular Boosting Models",
        "Input: 41 behavioral features\n\nRandomForest\nXGBoost\nLightGBM",
        "#D7F0EA",
    )
    add_box(
        ax,
        0.58,
        0.06,
        0.34,
        0.11,
        "6B. Temporal Sequence Models",
        "Convert sessions into sequences: (9000, 25, 15)\n\nCNN\nLSTM\nCNN-LSTM\nCNN-BiLSTM\nCNN-Attention-LSTM",
        "#DCE8FA",
    )

    add_box(
        ax,
        0.08,
        -0.08,
        0.34,
        0.10,
        "7. Model Evaluation",
        "Accuracy\nPrecision\nRecall\nF1 Score\nROC AUC\nConfusion Matrix",
        "#FFE5D0",
    )
    add_box(
        ax,
        0.58,
        -0.08,
        0.34,
        0.10,
        "8. Fraud Classification Output",
        "Predicted classes:\nHuman\nModerate Bot\nAdvanced Bot",
        "#E2F0CB",
    )

    add_arrow(ax, (0.25, 0.06), (0.25, 0.02))
    add_arrow(ax, (0.75, 0.06), (0.75, 0.02))
    add_arrow(ax, (0.42, -0.03), (0.58, -0.03))

    ax.text(
        0.5,
        -0.12,
        "Parallel learning branches share the same preprocessed behavioral dataset,\nwhile deep models additionally consume temporal session sequences.",
        ha="center",
        va="center",
        fontsize=11,
        color="#486581",
    )

    plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.22)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the project architecture diagram."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    build_matplotlib_diagram(OUTPUT_PATH)
    print("Architecture diagram generated successfully.")


if __name__ == "__main__":
    main()
