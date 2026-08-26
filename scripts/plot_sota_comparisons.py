"""
plot_sota_comparisons.py — Master Comparative Benchmark & Publication Figures
==============================================================================
Loads prediction CSVs from DeepSentinel and all competing baseline architectures
(MesoNet-4, XceptionNet, Multimodal ResNet-AV, and AceNet), computes full statistical
metrics (Acc, BalAcc, Prec, Rec, Spec, F1, MCC, AUC, DeLong p-value), and generates
high-resolution (300 DPI) publication comparison figures with inline Colab display.

Usage:
  python scripts/plot_sota_comparisons.py \
    --deepsentinel /content/drive/MyDrive/THESIS_MOTHERFILE/eval_results/fakeavceleb_eval_predictions.csv \
    --mesonet data/eval_results/preds_mesonet.csv \
    --xception data/eval_results/preds_xception.csv \
    --resnet_av data/eval_results/preds_resnet_av.csv \
    --output_dir data/eval_results/figures_comparative
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.significance import delong_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Premium Styling Palettes
COLORS = {
    "DeepSentinel (Ours)": "#0284c7",    # Electric Ocean Blue
    "MesoNet-4":           "#ef4444",    # Crimson Red
    "XceptionNet":         "#f59e0b",    # Amber Orange
    "Multimodal ResNet-AV":"#10b981",    # Emerald Green
    "AceNet (Baseline)":   "#8b5cf6",    # Purple
}


def load_model_predictions(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    recs = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["clip_id"]
            score = float(row["score"])
            pred = int(row.get("pred", 1 if score >= 0.50 else 0))
            recs[cid] = {
                "clip_id": cid,
                "fake_label": int(row["fake_label"]),
                "method": row.get("method", "unknown"),
                "type": row.get("type", "unknown"),
                "score": score,
                "pred": pred,
            }
    return recs


def compute_comprehensive_metrics(y_true: list[int], y_score: list[float], y_pred: list[int]) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = (tn / max(tn + fp, 1)) * 100.0
    bal_acc = (rec + spec) / 2.0

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.50

    # 1,000-Iteration Bootstrap for AUC 95% CI
    rng = np.random.default_rng(42)
    n = len(y_true)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true[idx])) == 2:
            try:
                boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
            except Exception:
                pass
    auc_lo = np.percentile(boot_aucs, 2.5) if boot_aucs else auc
    auc_hi = np.percentile(boot_aucs, 97.5) if boot_aucs else auc

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "prec": prec,
        "rec": rec,
        "spec": spec,
        "f1": f1,
        "mcc": mcc,
        "auc": auc,
        "auc_lo": auc_lo,
        "auc_hi": auc_hi,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def print_master_academic_table(models_data: dict, ref_name: str = "DeepSentinel (Ours)"):
    print("\n" + "=" * 120)
    print("  🏆 MASTER COMPARATIVE BENCHMARK TABLE (RESEARCH & THESIS CHAPTER 4)")
    print("=" * 120)
    print(f"  {'Model / Framework':<24} {'Modality':<14} {'Acc (%)':>8} {'BalAcc':>8} {'Spec(Real)':>11} {'Rec(Fake)':>10} {'F1':>7} {'MCC':>8} {'AUC-ROC [95% CI]':>20} {'DeLong p':>10}")
    print("  " + "-" * 118)

    ref_preds = models_data[ref_name]
    for name, data in models_data.items():
        m = data["metrics"]
        modality = data.get("modality", "Visual")
        
        # Compute DeLong p-value against DeepSentinel
        if name == ref_name:
            p_val_str = "— (Ref)"
        else:
            try:
                shared = sorted(set(ref_preds["clips"]) & set(data["clips"]))
                y_s = [ref_preds["clips"][c]["fake_label"] for c in shared]
                sc_ref = [ref_preds["clips"][c]["score"] for c in shared]
                sc_mod = [data["clips"][c]["score"] for c in shared]
                res = delong_test(y_s, sc_ref, sc_mod)
                p_val_str = f"{res.p_value:.4f}*" if res.p_value < 0.05 else f"{res.p_value:.4f}"
            except Exception:
                p_val_str = "N/A"

        auc_str = f"{m['auc']:.4f} [{m['auc_lo']:.3f}-{m['auc_hi']:.3f}]"
        print(f"  {name:<24} {modality:<14} {m['acc']:>7.2f}% {m['bal_acc']:>7.2f}% {m['spec']:>10.2f}% {m['rec']:>9.2f}% {m['f1']:>7.4f} {m['mcc']:>+7.4f} {auc_str:>20} {p_val_str:>10}")
    print("=" * 120 + "\n")


def plot_comparative_figures(models_data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11, "figure.dpi": 300})

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 1: Overlaid Multi-Model ROC Curves
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, data in models_data.items():
        y_true = data["y_true"]
        y_score = data["y_score"]
        color = COLORS.get(name, "#6b7280")
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = data["metrics"]["auc"]
        lw = 3.0 if "DeepSentinel" in name else 2.0
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", fontsize=12)
    ax.set_title("Cross-Dataset Generalization: Multi-Model ROC Comparison\n(Evaluated on Strictly Unseen FakeAVCeleb)", fontweight="bold", fontsize=13, pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10)
    
    fig1_path = out_dir / "comparative_roc_curves.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  [FIGURE 1] Saved Multi-Model ROC Curves -> {fig1_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 2: Multi-Metric Performance Bar Chart
    # ─────────────────────────────────────────────────────────────────────────
    metrics_names = ["Balanced Acc", "Real Specificity", "Fake Recall", "AUC-ROC (x100)"]
    n_models = len(models_data)
    x = np.arange(len(metrics_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, data) in enumerate(models_data.items()):
        m = data["metrics"]
        vals = [m["bal_acc"], m["spec"], m["rec"], m["auc"] * 100.0]
        color = COLORS.get(name, "#6b7280")
        offset = (i - n_models / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=name, color=color, edgecolor="black", linewidth=0.6, alpha=0.9)
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Performance (%)", fontweight="bold", fontsize=12)
    ax.set_title("Multi-Metric Benchmark Comparison Across Deepfake Detectors", fontweight="bold", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontweight="bold", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    fig2_path = out_dir / "comparative_multimetric_barchart.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  [FIGURE 2] Saved Multi-Metric Bar Chart -> {fig2_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 3: Per-Manipulation Stress Test Breakdown
    # ─────────────────────────────────────────────────────────────────────────
    all_methods = set()
    for data in models_data.values():
        for r in data["clips"].values():
            all_methods.add(r["method"])
    sorted_methods = sorted(all_methods)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_m = np.arange(len(sorted_methods))
    width_m = 0.18

    for i, (name, data) in enumerate(models_data.items()):
        accs = []
        for meth in sorted_methods:
            matching = [r for r in data["clips"].values() if r["method"] == meth]
            if matching:
                y_m = [r["fake_label"] for r in matching]
                p_m = [r["pred"] for r in matching]
                accs.append(sum(p == y for p, y in zip(p_m, y_m)) / len(matching) * 100.0)
            else:
                accs.append(0.0)
        
        color = COLORS.get(name, "#6b7280")
        offset = (i - n_models / 2 + 0.5) * width_m
        rects = ax.bar(x_m + offset, accs, width_m, label=name, color=color, edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold", fontsize=12)
    ax.set_title("Per-Manipulation Stress Breakdown Across All Detectors", fontweight="bold", fontsize=13, pad=12)
    ax.set_xticks(x_m)
    ax.set_xticklabels(sorted_methods, fontweight="bold", fontsize=10, rotation=15)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    fig3_path = out_dir / "comparative_method_breakdown.png"
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  [FIGURE 3] Saved Method Breakdown Bar Chart -> {fig3_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 4: Master 4-in-1 Comparative Thesis Dashboard
    # ─────────────────────────────────────────────────────────────────────────
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 13))

    # Subplot 1: ROC Curves
    for name, data in models_data.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_score"])
        ax1.plot(fpr, tpr, color=COLORS.get(name, "#6b7280"), lw=2.5 if "DeepSentinel" in name else 1.8,
                 label=f"{name} ({data['metrics']['auc']:.4f})")
    ax1.plot([0, 1], [0, 1], "--", color="#9ca3af")
    ax1.set_title("(A) Overlaid ROC Curves", fontweight="bold")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend(loc="lower right", fontsize=8)

    # Subplot 2: Multi-Metric Comparison
    for i, (name, data) in enumerate(models_data.items()):
        m = data["metrics"]
        vals = [m["bal_acc"], m["spec"], m["rec"], m["auc"] * 100.0]
        offset = (i - n_models / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width, label=name, color=COLORS.get(name, "#6b7280"), alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, fontsize=9, fontweight="bold")
    ax2.set_title("(B) Key Performance Metrics", fontweight="bold")
    ax2.set_ylabel("Score (%)")
    ax2.set_ylim(0, 110)
    ax2.legend(loc="upper left", fontsize=8)

    # Subplot 3: Method Breakdown
    for i, (name, data) in enumerate(models_data.items()):
        accs = []
        for meth in sorted_methods:
            matching = [r for r in data["clips"].values() if r["method"] == meth]
            accs.append(sum(r["pred"] == r["fake_label"] for r in matching) / max(len(matching), 1) * 100.0)
        offset = (i - n_models / 2 + 0.5) * width_m
        ax3.bar(x_m + offset, accs, width_m, label=name, color=COLORS.get(name, "#6b7280"), alpha=0.9)
    ax3.set_xticks(x_m)
    ax3.set_xticklabels(sorted_methods, fontsize=8, fontweight="bold", rotation=20)
    ax3.set_title("(C) Category-Wise Manipulation Accuracy", fontweight="bold")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(0, 110)

    # Subplot 4: F1 & Matthews Correlation Comparison
    mcc_f1_names = ["F1-Score", "Matthews Correlation (MCC)"]
    x_mf = np.arange(len(mcc_f1_names))
    for i, (name, data) in enumerate(models_data.items()):
        m = data["metrics"]
        vals = [m["f1"], m["mcc"]]
        offset = (i - n_models / 2 + 0.5) * width
        ax4.bar(x_mf + offset, vals, width, label=name, color=COLORS.get(name, "#6b7280"), alpha=0.9)
    ax4.set_xticks(x_mf)
    ax4.set_xticklabels(mcc_f1_names, fontsize=10, fontweight="bold")
    ax4.set_title("(D) Correlation & F1 Score (Parity Robustness)", fontweight="bold")
    ax4.set_ylabel("Score")
    ax4.set_ylim(-0.1, 1.05)
    ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle("DeepSentinel vs. Open-Source SOTA Baselines — Master Comparative Dashboard\n(FakeAVCeleb v1.2 Cross-Dataset Benchmark)",
                 fontweight="bold", fontsize=16, y=0.99)
    fig4_path = out_dir / "thesis_master_comparative_dashboard.png"
    plt.tight_layout()
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  [FIGURE 4] Saved 4-in-1 Master Comparative Dashboard -> {fig4_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Automatic Colab Display & Google Drive Backup
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from IPython.display import Image, display
        print("\n" + "=" * 60)
        print("  📊 RENDERING MASTER THESIS DASHBOARD IN NOTEBOOK:")
        print("=" * 60)
        display(Image(filename=str(fig4_path)))
    except Exception:
        pass

    folder_name = out_dir.name
    drive_backup_dirs = [
        Path(f"/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results/{folder_name}"),
        Path(f"/content/drive/MyDrive/eval_results/{folder_name}"),
    ]
    for d_dir in drive_backup_dirs:
        try:
            d_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for f in [fig1_path, fig2_path, fig3_path, fig4_path]:
                shutil.copy(f, d_dir / f.name)
            print(f"  [GOOGLE DRIVE BACKUP] Synced comparative figures directly -> {d_dir}")
            break
        except Exception:
            pass


def resolve_deepsentinel_path(ds_arg: str, split_arg: str | None = None) -> Path:
    target = split_arg if split_arg else ds_arg
    candidates = []
    
    if target in ["5000", "350+4650", "large"]:
        candidates = [
            Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results/fakeavceleb_eval_predictions_350+4650.csv"),
            Path("/content/drive/MyDrive/eval_results/fakeavceleb_eval_predictions_350+4650.csv"),
            REPO_ROOT / "data/eval_results/fakeavceleb_eval_predictions_350+4650.csv",
        ]
    elif target in ["700", "350+350", "balanced"]:
        candidates = [
            Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results/fakeavceleb_eval_predictions_350+350.csv"),
            Path("/content/drive/MyDrive/eval_results/fakeavceleb_eval_predictions_350+350.csv"),
            REPO_ROOT / "data/eval_results/fakeavceleb_eval_predictions_350+350.csv",
        ]
    else:
        p = Path(target)
        if p.exists():
            return p
        candidates = [
            p,
            Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results") / p.name,
            Path("/content/drive/MyDrive/eval_results") / p.name,
            Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results/fakeavceleb_eval_predictions_350+4650.csv"),
            Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results/fakeavceleb_eval_predictions.csv"),
            REPO_ROOT / "data/eval_results" / p.name,
        ]

    for cand in candidates:
        if cand.exists():
            print(f"  [Auto-Resolve] Discovered DeepSentinel predictions at: {cand}")
            return cand

    return Path(ds_arg)


def main():
    parser = argparse.ArgumentParser(description="Master Comparative Evaluation & Visualizations")
    parser.add_argument("--deepsentinel", type=str, default="fakeavceleb_eval_predictions_350+4650.csv",
                        help="DeepSentinel evaluation predictions CSV")
    parser.add_argument("--split", type=str, default=None, choices=["5000", "700", "350+4650", "350+350"],
                        help="Split selector: '5000' or '700'")
    parser.add_argument("--mesonet", type=str, default="data/eval_results/preds_mesonet.csv",
                        help="MesoNet-4 predictions CSV")
    parser.add_argument("--xception", type=str, default="data/eval_results/preds_xception.csv",
                        help="Xception predictions CSV")
    parser.add_argument("--resnet_av", type=str, default="data/eval_results/preds_resnet_av.csv",
                        help="Multimodal ResNet-AV predictions CSV")
    parser.add_argument("--acenet", type=str, default="data/eval_results/preds_acenet.csv",
                        help="Replicated AceNet predictions CSV")
    parser.add_argument("--output_dir", type=str, default="data/eval_results/figures_comparative",
                        help="Directory to save publication figures")
    args = parser.parse_args()

    ds_path = resolve_deepsentinel_path(args.deepsentinel, args.split)

    models_config = [
        ("DeepSentinel (Ours)", ds_path, "Affect-Bilinear"),
        ("MesoNet-4", Path(args.mesonet), "Visual CNN"),
        ("XceptionNet", Path(args.xception), "Visual Spatial"),
        ("Multimodal ResNet-AV", Path(args.resnet_av), "Audio-Visual"),
        ("AceNet (Baseline)", Path(args.acenet), "Cross-Attention"),
    ]

    models_data = {}
    for name, p, modality in models_config:
        clips = load_model_predictions(p)
        if clips:
            y_true = [r["fake_label"] for r in clips.values()]
            y_score = [r["score"] for r in clips.values()]
            y_pred = [r["pred"] for r in clips.values()]
            metrics = compute_comprehensive_metrics(y_true, y_score, y_pred)
            models_data[name] = {
                "clips": clips,
                "y_true": y_true,
                "y_score": y_score,
                "y_pred": y_pred,
                "metrics": metrics,
                "modality": modality,
            }

    if not models_data:
        print("ERROR: No prediction CSV files found! Run eval_public_baselines.py first.")
        return

    print_master_academic_table(models_data, ref_name="DeepSentinel (Ours)" if "DeepSentinel (Ours)" in models_data else list(models_data.keys())[0])
    plot_comparative_figures(models_data, Path(args.output_dir))


if __name__ == "__main__":
    main()
