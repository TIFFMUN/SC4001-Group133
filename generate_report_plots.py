"""
generate_report_plots.py
-------------------------
Generates publication-grade figures for the SC4001 Group 133 report:
1. Figure 1: Training and Validation Loss Convergence (from trainer_state.json)
2. Figure 2: The Exp 6 Budget Router Pareto Frontier (from exp6_router_sweep.csv)
3. Figure 3: Comparative Strategy and Scaling Performance (Exp 0, 1, 2, 3, 4, 5)
4. Figure 4: Qualitative Failure Case Taxonomy Breakdown
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set clean academic styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

base_dir = Path(__file__).parent.resolve()
results_dir = base_dir / "results"
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)


def plot_training_loss():
    trainer_state_path = (
        results_dir / "checkpoints" / "lora_r16" / "checkpoint-1263" / "trainer_state.json"
    )
    if not trainer_state_path.exists():
        print(f"[!] Warning: {trainer_state_path} not found.")
        return

    with open(trainer_state_path) as f:
        data = json.load(f)

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in data.get("log_history", []):
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=300)
    ax.plot(train_steps, train_loss, color="#1B365D", label="Training Loss", linewidth=1.8)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, color="#E05A47", marker="o", markersize=5,
                label="Validation Loss", linewidth=1.8, linestyle="--")

    ax.set_title("LoRA SFT Training & Validation Loss (Qwen2.5-0.5B, r=16)")
    ax.set_xlabel("Optimization Steps (3 Epochs, Effective Batch Size = 16)")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()

    out_file = plots_dir / "fig1_training_loss.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_pareto_frontier():
    sweep_path = results_dir / "exp6_router_sweep.csv"
    if not sweep_path.exists():
        print(f"[!] Warning: {sweep_path} not found.")
        return

    df = pd.read_csv(sweep_path)

    # Convert latencies to seconds
    df["latency_s"] = df["mean_latency_ms"] / 1000.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=300)

    # Subplot 1: Accuracy vs Latency
    pareto_df = df[df["pareto_optimal"] == True]
    non_pareto_df = df[df["pareto_optimal"] == False]

    ax1.plot(pareto_df["latency_s"], pareto_df["accuracy"] * 100, color="#1B365D",
             linestyle="-", linewidth=1.8, label="Pareto Frontier")
    ax1.scatter(pareto_df["latency_s"], pareto_df["accuracy"] * 100, color="#2E7D32",
                s=45, zorder=5, label="Pareto Optimal")
    ax1.scatter(non_pareto_df["latency_s"], non_pareto_df["accuracy"] * 100, color="#D32F2F",
                marker="x", s=40, zorder=4, label="Suboptimal")

    ax1.set_title("Accuracy vs. Generation Latency")
    ax1.set_xlabel("Mean Latency per Query (seconds)")
    ax1.set_ylabel("GSM8K Accuracy (%)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Subplot 2: Accuracy vs Token Budget
    ax2.plot(pareto_df["mean_token_count"], pareto_df["accuracy"] * 100, color="#0F2042",
             linestyle="-", linewidth=1.8, label="Pareto Frontier")
    ax2.scatter(pareto_df["mean_token_count"], pareto_df["accuracy"] * 100, color="#2E7D32",
                s=45, zorder=5)
    ax2.scatter(non_pareto_df["mean_token_count"], non_pareto_df["accuracy"] * 100, color="#D32F2F",
                marker="x", s=40, zorder=4)

    ax2.set_title("Accuracy vs. Mean Token Budget")
    ax2.set_xlabel("Mean Generated Tokens per Query")
    ax2.set_ylabel("GSM8K Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = plots_dir / "fig2_pareto_frontier.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_strategy_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)

    # Panel A: Phase 1 (100 samples, 256 tokens) vs Phase 2 (500 samples, 600 tokens)
    categories = ["Exp 0: Zero-Shot", "Exp 2: LoRA Adaptation"]
    acc_100 = [29.0, 35.0]
    acc_500 = [46.0, 31.2]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, acc_100, width, label="Phase 1: 100 samples (256 tok cap)",
                     color="#90A4AE", edgecolor="#222222", linewidth=0.8)
    rects2 = ax1.bar(x + width/2, acc_500, width, label="Phase 2: 500 samples (600 tok budget)",
                     color="#1B365D", edgecolor="#222222", linewidth=0.8)

    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.8, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.8, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Impact of Token Budget & Sample Scaling")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 55)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel B: Full Scale Comparison Across All Strategies (500+ samples, 600 tokens)
    strats = [
        "Exp 0\n(Zero-Shot)",
        "Exp 2\n(LoRA SFT)",
        "Exp 4\n(Distill Filt.)",
        "Exp 3\n(Distill All)",
        "Exp 5\n(ARC Transfer)"
    ]
    accs = [46.00, 31.20, 35.80, 40.80, 33.62]
    colors = ["#64B5F6", "#E05A47", "#81C784", "#2E7D32", "#FFB300"]

    bars = ax2.bar(strats, accs, color=colors, width=0.55, edgecolor="#222222", linewidth=0.8)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, h + 0.8, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(b) Full Strategy Benchmark (500+ samples, 600 tok)")
    ax2.set_ylim(0, 55)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = plots_dir / "fig3_strategy_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_failure_taxonomy():
    labels = [
        "Truncation Collapse\n(Artificial Token Cap)",
        "Arithmetic Calculation\nSlips",
        "Entity / Premise\nConfusion"
    ]
    sizes = [45, 35, 20]
    colors = ["#E05A47", "#1B365D", "#FFB300"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%", startangle=140,
        colors=colors, textprops=dict(color="#222222", fontsize=9),
        wedgeprops=dict(width=0.45, edgecolor="#FFFFFF", linewidth=1.5)
    )

    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title("Qualitative Failure Mode Taxonomy (GSM8K Errors)")
    plt.tight_layout()

    out_file = plots_dir / "fig4_error_taxonomy.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


if __name__ == "__main__":
    plot_training_loss()
    plot_pareto_frontier()
    plot_strategy_comparison()
    plot_failure_taxonomy()
    print("[+] All publication figures generated successfully in results/plots/!")
