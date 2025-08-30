#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate paper plots from a metrics file.

Generates three figures:
  1) Target performance vs epoch (accuracy + macro-F1)
  2) Final per-class recall (grouped bars)
  3) Domain gap vs epoch (method-appropriate metric, normalized to epoch 1)

Usage:
  python make_plots.py --metrics /path/to/all_methods_10epochs.csv --out outputs/figs --small

Notes:
  - Uses matplotlib only (no seaborn; no custom colors).
  - Saves PDFs with compact 'small' variants for single-column layout.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def _savefig(fig, path: Path):
    fig.tight_layout(pad=0.6)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def plot_target_performance_vs_epoch(df: pd.DataFrame, outdir: Path, small: bool = False) -> Path:
    """Plot target accuracy and macro-F1 vs epoch for each method."""
    agg = (
        df.groupby(["method", "epoch"])
          .agg(target_acc=("target_acc", "mean"),
               target_macro_f1=("target_macro_f1", "mean"))
          .reset_index()
          .sort_values(["method", "epoch"])
    )

    # Figure size
    figsize = (3.3, 2.2) if small else (7.0, 4.0)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.gca()

    handles, labels = [], []
    for m, sub in agg.groupby("method"):
        sub = sub.sort_values("epoch")
        h_acc, = ax.plot(sub["epoch"], sub["target_acc"], linewidth=1.2, label=m)
        # Macro-F1 as dashed, scaled to %
        ax.plot(sub["epoch"], sub["target_macro_f1"] * 100.0, linestyle="--", linewidth=1.0)
        handles.append(h_acc); labels.append(m)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acc / F1 (%)")
    ax.set_title("Target Performance vs Epoch")
    ax.legend(handles, labels, loc="best", fontsize=8 if small else 9, ncol=1,
              framealpha=0.9, borderpad=0.3, handlelength=1.5, handletextpad=0.4)
    ax.text(0.98, 0.02, "Dashed = Macro-F1", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=6 if small else 8)

    name = "small_target_performance_vs_epoch.pdf" if small else "plot_target_performance_vs_epoch.pdf"
    out_path = outdir / name
    _savefig(fig, out_path)
    return out_path


def plot_per_class_recall_final(df: pd.DataFrame, outdir: Path, small: bool = False) -> Path:
    """Grouped bars of final-epoch per-class recall for each method."""
    finals = df.sort_values("epoch").groupby(["method"]).tail(1)

    methods = list(finals["method"].unique())
    class_cols = ["recall_elliptical", "recall_spiral", "recall_irregular"]

    # Build averaged table (handles potential multiple runs per method)
    rows = []
    for m in methods:
        sub = finals[finals["method"] == m]
        row = {"method": m}
        for c in class_cols:
            row[c] = float(sub[c].mean())
        rows.append(row)
    avg = pd.DataFrame(rows)

    # Figure
    figsize = (3.3, 2.2) if small else (7.0, 4.0)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.gca()

    x = np.arange(len(class_cols))
    width = 0.8 / max(1, len(methods))

    for j, m in enumerate(methods):
        vals = [100.0 * avg.loc[avg["method"] == m, c].values[0] for c in class_cols]
        ax.bar(x + j * width, vals, width, label=m)

    ax.set_xticks(x + width * (len(methods) - 1) / 2.0)
    xtlbl = ["Elliptical", "Spiral", "Irregular"]
    if small:
        xtlbl = ["E", "S", "I"]
    ax.set_xticklabels(xtlbl)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-Class Recall (Final)")
    ax.legend(loc="best", fontsize=8 if small else 9, ncol=1,
              framealpha=0.9, borderpad=0.3, handlelength=1.2, handletextpad=0.4)

    name = "small_per_class_recall_final.pdf" if small else "plot_per_class_recall_final.pdf"
    out_path = outdir / name
    _savefig(fig, out_path)
    return out_path


def plot_domain_gap_vs_epoch(df: pd.DataFrame, outdir: Path, small: bool = False) -> Path:
    """Plot normalized domain gap vs epoch, per method.
    For MMD: use mmd2
    For Sinkhorn: use sinkhorn_div
    For DANN: use |domain_auc - 0.5|
    Methods lacking a relevant metric are skipped.
    """
    agg = (
        df.groupby(["method", "epoch"])
          .agg(mmd2=("mmd2", "mean"),
               sinkhorn_div=("sinkhorn_div", "mean"),
               domain_auc=("domain_auc", "mean"))
          .reset_index()
          .sort_values(["method", "epoch"])
    )

    def gap_metric(row):
        m = str(row["method"]).lower()
        if m == "mmd":
            return row["mmd2"]
        if m == "sinkhorn":
            return row["sinkhorn_div"]
        if m == "dann":
            return abs(row["domain_auc"] - 0.5) if pd.notna(row["domain_auc"]) else np.nan
        return np.nan

    agg["gap_raw"] = agg.apply(gap_metric, axis=1)

    # Normalize gap by the first epoch per method
    agg["gap_norm"] = np.nan
    for m, sub in agg.groupby("method"):
        sub = sub.sort_values("epoch")
        base = sub["gap_raw"].iloc[0]
        if pd.notna(base) and base != 0:
            agg.loc[sub.index, "gap_norm"] = sub["gap_raw"] / base
        else:
            agg.loc[sub.index, "gap_norm"] = sub["gap_raw"]  # leave as-is

    # Figure
    figsize = (3.3, 2.2) if small else (7.0, 4.0)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.gca()

    any_plotted = False
    for m, sub in agg.groupby("method"):
        sub = sub.sort_values("epoch")
        if not sub["gap_norm"].notna().any():
            continue
        ax.plot(sub["epoch"], sub["gap_norm"], label=m, linewidth=1.2)
        any_plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized domain gap (↓)")
    ax.set_title("Domain Gap vs Epoch")
    if any_plotted:
        ax.legend(loc="best", fontsize=8 if small else 9,
                  framealpha=0.9, borderpad=0.3, handlelength=1.5, handletextpad=0.4)

    name = "small_domain_gap_vs_epoch.pdf" if small else "plot_domain_gap_vs_epoch.pdf"
    out_path = outdir / name
    _savefig(fig, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Recreate paper plots from metrics CSV.")
    ap.add_argument("--metrics", type=str, required=True,
                    help="Path to all_methods_10epochs.csv")
    ap.add_argument("--out", type=str, default="outputs/figs",
                    help="Output directory for figures")
    ap.add_argument("--small", action="store_true",
                    help="Also produce compact single-column variants")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    outdir = Path(args.out)
    _ensure_outdir(outdir)

    # Load
    df = pd.read_csv(metrics_path)
    # Ensure epoch numeric
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    # Plot standard-size figures
    p1 = plot_target_performance_vs_epoch(df, outdir, small=False)
    p2 = plot_per_class_recall_final(df, outdir, small=False)
    p3 = plot_domain_gap_vs_epoch(df, outdir, small=False)

    paths = [p1, p2, p3]

    # Optional small variants
    if args.small:
        sp1 = plot_target_performance_vs_epoch(df, outdir, small=True)
        sp2 = plot_per_class_recall_final(df, outdir, small=True)
        sp3 = plot_domain_gap_vs_epoch(df, outdir, small=True)
        paths.extend([sp1, sp2, sp3])

    # Console summary
    print("Saved figures:")
    for p in paths:
        print(" -", p)


if __name__ == "__main__":
    main()
