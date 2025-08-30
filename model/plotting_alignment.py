import numpy as np, pandas as pd, matplotlib.pyplot as plt

def _norm_start(df, col):
    # normalize each method's curve by its first-epoch value (for trend comparability)
    g = df.groupby("method")[col]
    base = g.transform(lambda s: s.iloc[0] if len(s)>0 else np.nan)
    return df[col] / (base.replace(0, np.nan))

def plot_training_and_alignment(df, class_names=("elliptical","irregular","spiral"),
                                savepath="fig_training_alignment.pdf"):
    methods = list(df["method"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axA, axB, axC, axD = axes.ravel()

    # (A) Target Acc + Macro-F1
    for m in methods:
        sub = df[df.method==m].sort_values("epoch")
        axA.plot(sub["epoch"], sub["target_acc"], label=f"{m} Acc")
        axA.plot(sub["epoch"], sub["target_macro_f1"]*100, linestyle="--", label=f"{m} Macro-F1 (%)")
    axA.set_xlabel("Epoch"); axA.set_ylabel("Target (%)"); axA.set_title("Target performance vs epoch")
    axA.legend(fontsize=8, ncol=2)

    # (B) Domain-gap vs epoch (method-appropriate metric)
    gap_vals = []
    for _, r in df.iterrows():
        m = r["method"].lower()
        if m=="mmd":
            gap_vals.append(r["mmd2"])
        elif m=="sinkhorn":
            gap_vals.append(r["sinkhorn_div"])
        elif m=="dann":
            # smaller (AUC closer to 0.5) is better; use |AUC-0.5|
            gap_vals.append(abs(r["domain_auc"] - 0.5))
        else:
            gap_vals.append(np.nan)
    df = df.copy()
    df["gap_raw"] = gap_vals
    df["gap_norm"] = _norm_start(df, "gap_raw")

    for m in methods:
        sub = df[df.method==m].sort_values("epoch")
        if sub["gap_norm"].notna().any():
            axB.plot(sub["epoch"], sub["gap_norm"], label=m)
    axB.set_xlabel("Epoch"); axB.set_ylabel("Normalized gap"); axB.set_title("Domain gap vs epoch (↓ better)")
    axB.legend(fontsize=8)

    # (C) Class-conditional alignment at final epoch (method-specific bars)
    finals = df.sort_values("epoch").groupby("method").tail(1)
    x = np.arange(len(class_names))
    width = 0.8 / max(1, len(methods))

    for j, m in enumerate(methods):
        row = finals[finals.method==m]
        if row.empty: continue
        row = row.iloc[0]
        mlow = m.lower()
        if mlow=="mmd":
            vals = [row.get(f"cmmd_{c}", np.nan) for c in class_names]
            axC.bar(x + j*width, vals, width, label=m)
            axC.set_ylabel("Class MMD$^2$ (↓)")
        elif mlow=="sinkhorn":
            # use diagonal entries of class–class OT mass if available
            vals = []
            if isinstance(row.get("ot_on_diag", np.nan), float) and np.isnan(row.get("ot_on_diag", np.nan)):
                vals = [np.nan]*len(class_names)
            else:
                # we don't store the matrix in df; so show the global diag share per class via domAUC fallback
                vals = [np.nan]*len(class_names)
            axC.bar(x + j*width, vals, width, label=m)
            axC.set_ylabel("OT diag mass (↑)")
        elif mlow=="dann":
            # plot (0.5 - domAUC_class). Larger is better (→ 0.5)
            vals=[]
            for c in class_names:
                auc = row.get(f"domAUC_{c}", np.nan)
                vals.append(0.5 - abs(auc-0.5) if np.isfinite(auc) else np.nan)
            axC.bar(x + j*width, vals, width, label=m)
            axC.set_ylabel("Domain indistinguishability (↑)")
        else:
            vals = [np.nan]*len(class_names)
            axC.bar(x + j*width, vals, width, label=m)

    axC.set_xticks(x + width*(len(methods)-1)/2)
    axC.set_xticklabels([c.title() for c in class_names])
    axC.set_title("Class-conditional alignment (final epoch)")
    axC.legend(fontsize=8, ncol=2)

    # (D) Losses vs epoch
    for m in methods:
        sub = df[df.method==m].sort_values("epoch")
        axD.plot(sub["epoch"], sub["train_ce_loss"], label=f"{m} CE")
        if sub["train_aux_loss"].notna().any():
            axD.plot(sub["epoch"], sub["train_aux_loss"], linestyle="--", label=f"{m} aux")
    axD.set_xlabel("Epoch"); axD.set_ylabel("Loss"); axD.set_title("Loss terms vs epoch")
    axD.legend(fontsize=8, ncol=2)

    fig.tight_layout(); fig.savefig(savepath, bbox_inches="tight")
    print(f"Saved {savepath}")

def plot_per_class_recalls(df, class_names=("elliptical","irregular","spiral"), savepath="fig_perclass.pdf"):
    methods = list(df["method"].unique())
    finals = df.sort_values("epoch").groupby("method").tail(1)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(class_names)); width = 0.8 / max(1, len(methods))
    for j, m in enumerate(methods):
        row = finals[finals.method==m]
        if row.empty: continue
        row = row.iloc[0]
        vals = [100.0*row.get(f"recall_{c}", np.nan) for c in class_names]
        ax.bar(x + j*width, vals, width, label=m)
    ax.set_xticks(x + width*(len(methods)-1)/2)
    ax.set_xticklabels([c.title() for c in class_names])
    ax.set_ylabel("Recall (%)"); ax.set_title("Target per-class recall (final epoch)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(savepath, bbox_inches="tight")
    print(f"Saved {savepath}")

def plot_embeddings(embeds, savepath="fig_embeddings.pdf", class_names=("elliptical","irregular","spiral")):
    keys = list(embeds.keys())
    if not keys: 
        print("No embeddings provided."); return
    # If you pass {"NoAdapt": ..., "MMD": ...} we show both; otherwise show the first twice.
    left_key = "NoAdapt" if "NoAdapt" in embeds else keys[0]
    best_key = keys[-1]

    panels = [(left_key, "domain"), (best_key, "domain"), (best_key, "class")]
    titles = [f"{left_key} — domain", f"{best_key} — domain", f"{best_key} — class"]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, (k, mode), title in zip(axes, panels, titles):
        E = embeds.get(k)
        if not E:
            ax.text(0.5, 0.5, f"No embedding for {k}", ha="center"); continue
        XY, dom, lab = E["xy"], E["domain"], E["label"]
        if mode=="domain":
            ms = (dom==0); mt=(dom==1)
            ax.scatter(XY[ms,0], XY[ms,1], s=3, alpha=0.6, label="source")
            ax.scatter(XY[mt,0], XY[mt,1], s=3, alpha=0.6, label="target")
            ax.legend(fontsize=8)
        else:
            for i, cname in enumerate(class_names):
                m = (lab==i)
                if m.sum()==0: continue
                ax.scatter(XY[m,0], XY[m,1], s=3, alpha=0.6, label=cname)
            ax.legend(fontsize=8, ncol=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    fig.tight_layout(); fig.savefig(savepath, bbox_inches="tight")
    print(f"Saved {savepath}")
