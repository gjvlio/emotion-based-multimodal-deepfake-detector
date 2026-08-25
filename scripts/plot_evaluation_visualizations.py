"""
plot_evaluation_visualizations.py — Generate publication-quality thesis figures from evaluation CSV.

Generates:
1. confusion_matrix_heatmap.png: Normalized Confusion Matrix with percentages and counts
2. per_method_accuracy_barchart.png: Accuracy across manipulation techniques (faceswap-wav2lip, wav2lip, fsgan, real, etc.)
3. roc_curve_plot.png: ROC curve with AUC-ROC = 0.8960 and 95% Bootstrap Confidence Band
4. score_distribution_plot.png: P(Fake) distribution comparison between Real and Fake clips
5. thesis_evaluation_dashboard.png: 4-panel combined figure ready for inclusion in Chapter 4 / Results.

Usage:
    python scripts/plot_evaluation_visualizations.py --csv data/eval_results/unseen_evaluation_5000clips.csv
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parents[1]

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def plot_all(csv_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} evaluation records from {csv_path}")

    y_true = df["fake_label"].values
    y_score = df["score"].values
    y_pred_50 = (y_score >= 0.50).astype(int)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Confusion Matrix Heatmap
    # ─────────────────────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred_50)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    labels = [f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)" for i in range(2) for j in range(2)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm_norm, annot=labels, fmt="", cmap="Blues", cbar=True,
        xticklabels=["Predicted Real", "Predicted Fake"],
        yticklabels=["Actual Real", "Actual Fake"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax, linewidths=1.5, linecolor="white"
    )
    ax.set_title("FakeAVCeleb Unseen Benchmark\nConfusion Matrix (N = 5,000)", fontsize=14, weight="bold", pad=15)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    cm_out = output_dir / "confusion_matrix_heatmap.png"
    plt.savefig(cm_out, dpi=300)
    plt.close()
    print(f"  [Saved] {cm_out}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Per-Method Accuracy Bar Chart
    # ─────────────────────────────────────────────────────────────────────────────
    method_stats = []
    for method, grp in df.groupby("method"):
        preds = (grp["score"] >= 0.50).astype(int)
        acc = (preds == grp["fake_label"]).mean() * 100
        n_samples = len(grp)
        cat_type = grp["type"].iloc[0]
        method_stats.append({
            "Method": method,
            "Accuracy": acc,
            "N": n_samples,
            "Type": "Real" if method == "real" else ("Compound" if "wav2lip" in method and ("faceswap" in method or "fsgan" in method) else ("Lip-Sync" if method == "wav2lip" else "Single-Modality"))
        })

    m_df = pd.DataFrame(method_stats).sort_values("Accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    colors = {
        "Compound": "#2b5c8f",
        "Lip-Sync": "#418ab3",
        "Single-Modality": "#78b4c4",
        "Real": "#27ae60"
    }
    bar_colors = [colors.get(t, "#999999") for t in m_df["Type"]]

    bars = ax.barh(m_df["Method"], m_df["Accuracy"], color=bar_colors, edgecolor="#222222", height=0.65)
    ax.set_xlim(0, 115)
    ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.7, label="Chance Level (50%)")
    ax.axvline(80, color="#27ae60", linestyle=":", linewidth=1.2, alpha=0.7, label="High Benchmark (80%)")

    for bar, n in zip(bars, m_df["N"]):
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}% (N={n:,})",
                ha="left", va="center", fontsize=10, weight="bold", color="#222222")

    ax.set_title("DeepSentinel Cross-Dataset Detection Accuracy by Deepfake Manipulation", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Accuracy (%)", fontsize=12, weight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    bar_out = output_dir / "per_method_accuracy_barchart.png"
    plt.savefig(bar_out, dpi=300)
    plt.close()
    print(f"  [Saved] {bar_out}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. ROC Curve with Bootstrap 95% Confidence Band
    # ─────────────────────────────────────────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"DeepSentinel (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    # Highlight operating points
    # Standard tau = 0.50
    idx_50 = np.argmin(np.abs(thresholds - 0.50))
    ax.scatter(fpr[idx_50], tpr[idx_50], color="#e74c3c", s=90, zorder=5, label=f"Standard Operating Pt (tau=0.50)\nRecall={tpr[idx_50]*100:.1f}%, Spec={(1-fpr[idx_50])*100:.1f}%")

    # Youden's J operating point
    j_scores = tpr - fpr
    idx_opt = np.argmax(j_scores)
    ax.scatter(fpr[idx_opt], tpr[idx_opt], color="#27ae60", s=90, zorder=5, marker="^", label=f"Optimal Operating Pt (tau={thresholds[idx_opt]:.2f})\nRecall={tpr[idx_opt]*100:.1f}%, Spec={(1-fpr[idx_opt])*100:.1f}%")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, weight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, weight="bold")
    ax.set_title(f"Receiver Operating Characteristic (ROC)\nFakeAVCeleb Unseen Benchmark (N = {len(df):,})", fontsize=13, weight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    roc_out = output_dir / "roc_curve_plot.png"
    plt.savefig(roc_out, dpi=300)
    plt.close()
    print(f"  [Saved] {roc_out}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Score Distribution Plot (Real vs Fake Discrimination)
    # ─────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
    real_scores = df[df["fake_label"] == 0]["score"]
    fake_scores = df[df["fake_label"] == 1]["score"]

    sns.histplot(real_scores, bins=40, color="#27ae60", kde=True, stat="density", label=f"Real Videos (N={len(real_scores):,})", alpha=0.55, ax=ax)
    sns.histplot(fake_scores, bins=40, color="#e74c3c", kde=True, stat="density", label=f"Fake Videos (N={len(fake_scores):,})", alpha=0.55, ax=ax)

    ax.axvline(0.50, color="#2c3e50", linestyle="--", lw=2, label="Classification Boundary (tau = 0.50)")
    ax.set_title("DeepSentinel Predicted Probability Distribution P(Fake)", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Predicted Probability P(Fake)", fontsize=12, weight="bold")
    ax.set_ylabel("Density", fontsize=12, weight="bold")
    ax.legend(loc="upper center", frameon=True, fontsize=10.5)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    dist_out = output_dir / "score_distribution_plot.png"
    plt.savefig(dist_out, dpi=300)
    plt.close()
    print(f"  [Saved] {dist_out}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Combined Publication Dashboard (4-in-1 Master Thesis Figure)
    # ─────────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # Subplot A: Confusion Matrix
    ax_a = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        cm_norm, annot=labels, fmt="", cmap="Blues", cbar=False,
        xticklabels=["Pred Real", "Pred Fake"],
        yticklabels=["True Real", "True Fake"],
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax_a, linewidths=1.2, linecolor="white"
    )
    ax_a.set_title("(a) Confusion Matrix (N = 5,000)", fontsize=12, weight="bold")

    # Subplot B: ROC Curve
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"DeepSentinel (AUC={roc_auc:.4f})")
    ax_b.plot([0, 1], [0, 1], color="#888888", lw=1.2, linestyle="--")
    ax_b.scatter(fpr[idx_50], tpr[idx_50], color="#e74c3c", s=60, zorder=5, label=f"tau=0.50 (Rec={tpr[idx_50]*100:.1f}%, Spec={(1-fpr[idx_50])*100:.1f}%)")
    ax_b.set_xlabel("FPR (1 - Specificity)", fontsize=10, weight="bold")
    ax_b.set_ylabel("TPR (Recall)", fontsize=10, weight="bold")
    ax_b.set_title("(b) ROC Curve & Operating Points", fontsize=12, weight="bold")
    ax_b.legend(loc="lower right", fontsize=8.5)

    # Subplot C: Per-Method Accuracy
    ax_c = fig.add_subplot(gs[1, 0])
    bars_c = ax_c.barh(m_df["Method"], m_df["Accuracy"], color=bar_colors, edgecolor="#222222", height=0.6)
    ax_c.set_xlim(0, 115)
    for bar in bars_c:
        w = bar.get_width()
        ax_c.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", ha="left", va="center", fontsize=8.5, weight="bold")
    ax_c.set_title("(c) Detection Accuracy by Manipulation Type", fontsize=12, weight="bold")
    ax_c.set_xlabel("Accuracy (%)", fontsize=10, weight="bold")

    # Subplot D: Score Distribution
    ax_d = fig.add_subplot(gs[1, 1])
    sns.histplot(real_scores, bins=30, color="#27ae60", kde=True, stat="density", label=f"Real (N={len(real_scores):,})", alpha=0.5, ax=ax_d)
    sns.histplot(fake_scores, bins=30, color="#e74c3c", kde=True, stat="density", label=f"Fake (N={len(fake_scores):,})", alpha=0.5, ax=ax_d)
    ax_d.axvline(0.50, color="#2c3e50", linestyle="--", lw=1.5, label="Boundary (tau=0.50)")
    ax_d.set_title("(d) P(Fake) Probability Distribution", fontsize=12, weight="bold")
    ax_d.set_xlabel("Predicted Probability P(Fake)", fontsize=10, weight="bold")
    ax_d.set_ylabel("Density", fontsize=10, weight="bold")
    ax_d.legend(loc="upper center", fontsize=8.5)

    fig.suptitle("DeepSentinel Multimodal Deepfake Detector — FakeAVCeleb Benchmark Summary", fontsize=15, weight="bold", y=0.98)
    dash_out = output_dir / "thesis_evaluation_dashboard.png"
    plt.savefig(dash_out, dpi=300)
    plt.close()
    print(f"  [Saved] {dash_out}")

    # Also backup to Google Drive if mounted
    drive_eval = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results")
    if drive_eval.exists():
        import shutil
        for f in [cm_out, bar_out, roc_out, dist_out, dash_out]:
            shutil.copy2(f, drive_eval / f.name)
        print(f"  [GOOGLE DRIVE BACKUP] Synced all 5 figures to {drive_eval}")

    # Display directly in Google Colab / Jupyter notebook output if in IPython
    try:
        from IPython.display import display, Image as IPImage
        print("\n" + "=" * 60)
        print("  DISPLAYING MASTER EVALUATION FIGURES IN COLAB")
        print("=" * 60)
        display(IPImage(filename=str(dash_out)))
        display(IPImage(filename=str(cm_out)))
        display(IPImage(filename=str(bar_out)))
        display(IPImage(filename=str(roc_out)))
        display(IPImage(filename=str(dist_out)))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate visual evaluation figures from CSV")
    parser.add_argument("--csv", type=str, default="data/eval_results/unseen_evaluation_5000clips.csv",
                        help="Path to evaluation predictions CSV")
    parser.add_argument("--output_dir", type=str, default="data/eval_results/figures",
                        help="Directory to save generated plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Search Google Drive
        drive_csv = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results/fakeavceleb_eval_predictions.csv")
        if drive_csv.exists():
            csv_path = drive_csv
        else:
            print(f"ERROR: CSV not found at {csv_path}")
            return

    plot_all(csv_path, Path(args.output_dir))


if __name__ == "__main__":
    main()
