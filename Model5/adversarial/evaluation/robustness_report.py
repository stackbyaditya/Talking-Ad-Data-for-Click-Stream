"""Plotting and report-generation helpers for Model5 robustness experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_epsilon_curve(curves: dict[str, list[dict[str, float]]], output_path: Path, metric_name: str = "accuracy") -> None:
    """Plot metric values over epsilon for multiple models."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, points in curves.items():
        frame = pd.DataFrame(points)
        ax.plot(frame["epsilon"], frame[metric_name], marker="o", linewidth=2, label=model_name)
    ax.set_title(f"Epsilon vs {metric_name.replace('_', ' ').title()}")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_clean_vs_attacked_vs_hardened(comparison: dict[str, dict[str, float]], output_path: Path) -> None:
    """Plot clean, attacked, and hardened accuracy for each model."""
    rows = []
    for model_name, metrics in comparison.items():
        for stage_name, value in metrics.items():
            rows.append({"model": model_name, "stage": stage_name, "accuracy": value})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(10, len(comparison) * 1.2), 6))
    sns.barplot(data=frame, x="model", y="accuracy", hue="stage", palette="Set2", ax=ax)
    ax.set_title("Clean vs Attacked vs Hardened Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_defense_recovery(defense_rows: list[dict[str, float]], output_path: Path) -> None:
    """Plot defense recovery on adversarial accuracy."""
    frame = pd.DataFrame(defense_rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=frame, x="model", y="defense_recovery", hue="defense", palette="Set1", ax=ax)
    ax.set_title("Defense Recovery on Adversarial Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy Recovery")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_report(report: dict[str, object], output_path: Path) -> None:
    """Save a JSON robustness report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_markdown_report(report: dict[str, object], output_path: Path) -> None:
    """Write a compact markdown summary of robustness results."""
    lines = [
        "# Model5 Robustness Summary",
        "",
        "## Baseline vs Adversarial",
    ]

    for family_name in ("tabular", "sequence"):
        family = report.get(family_name, {})
        if not family:
            continue
        lines.append(f"### {family_name.title()} Models")
        lines.append("| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy | FGSM ASR | PGD ASR |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for model_name, metrics in family.items():
            fgsm = metrics.get("fgsm", {})
            pgd = metrics.get("pgd", {})
            lines.append(
                f"| {model_name} | {metrics['clean']['accuracy']:.4f} | "
                f"{fgsm.get('accuracy', 0.0):.4f} | {pgd.get('accuracy', 0.0):.4f} | "
                f"{fgsm.get('asr', 0.0):.4f} | {pgd.get('asr', 0.0):.4f} |"
            )
        lines.append("")

    defenses = report.get("defenses", {})
    if defenses:
        lines.extend(
            [
                "## Defenses",
                "| Model | Defense | Clean Accuracy | Adversarial Accuracy | Hardened Accuracy | Recovery |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name, metrics in defenses.items():
            lines.append(
                f"| {model_name} | {metrics['defense']} | {metrics['clean_accuracy']:.4f} | "
                f"{metrics['adversarial_accuracy']:.4f} | {metrics['hardened_accuracy']:.4f} | {metrics['defense_recovery']:.4f} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
