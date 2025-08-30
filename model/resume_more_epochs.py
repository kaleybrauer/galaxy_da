import torch, pandas as pd
from pathlib import Path

# your pieces
from training_with_logging import train_with_lambda_logging
from save_runs import save_end_of_run  
from model import CNN, give_labels

device = "cuda" if torch.cuda.is_available() else "cpu"
label2idx, idx2label = give_labels()
class_names = tuple([c for c,_ in sorted(label2idx.items(), key=lambda kv: kv[1])])

def load_model_from_ckpt(base_model_ctor, ckpt_path, map_location="cpu"):
    """Return a model with weights loaded from a checkpoint that stored ONLY model.state_dict()."""
    model = base_model_ctor()
    state = torch.load(ckpt_path, map_location=map_location)
    # If you saved a full dict later, support both formats transparently
    state_dict = state.get("model_state") if isinstance(state, dict) and "model_state" in state else state
    model.load_state_dict(state_dict)
    return model

def make_model_ctor_from_ckpt(ckpt_path):
    base_ctor = lambda: CNN(num_classes=len(class_names)).to(device)
    return lambda: load_model_from_ckpt(base_ctor, ckpt_path, map_location=device)

optimizer_ctor = lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

def run_more(method_name, lambda_val, ckpt_path, train_loader_source, train_loader_target, val_loader_source,val_loader_target, blur=0.5, extra_epochs=5):
    """
    Resume training for +extra_epochs starting from the saved model weights.
    Note: optimizer/discriminator are re-initialized; that’s usually fine.
    """
    assert Path(ckpt_path).exists(), f"Missing checkpoint: {ckpt_path}"

    df_more, model_more = train_with_lambda_logging(
        lambda_val=lambda_val,
        method_name=method_name,
        model_ctor=make_model_ctor_from_ckpt(ckpt_path),
        optimizer_ctor=optimizer_ctor,
        train_loader_source=train_loader_source,
        train_loader_target=train_loader_target,
        eval_loader_source=val_loader_source,
        eval_loader_target=val_loader_target,
        device=device,
        num_epochs=extra_epochs,
        blur=blur,
        class_names=class_names,
    )
    # Save metrics + final weights (new files, won’t overwrite originals)
    out = save_end_of_run(df_more, model_more, f"{method_name}_resume", lambda_val)
    return df_more, out

def build_master_10epochs(
    initial_csv: str,                 # e.g. "logs/all_methods_metrics_latest.csv"
    resume_csv: str,                  # e.g. "logs/all_methods_metrics_resume.csv"
    out_csv: str = "logs/all_methods_10epochs.csv"
) -> pd.DataFrame:
    """Merge initial (epochs 1..N) + resume (epochs 1..M) into a master file with epoch -> 1..(N+M) per method."""
    init = pd.read_csv(initial_csv).copy()
    res  = pd.read_csv(resume_csv).copy()

    # Basic hygiene
    for df, phase in [(init, "initial"), (res, "resume")]:
        if "method" not in df or "epoch" not in df:
            raise ValueError(f"{phase} CSV must contain columns 'method' and 'epoch'")
        df["phase"] = phase
        df["epoch"] = df["epoch"].astype(int)

    frames = []
    methods = sorted(set(init["method"]).union(set(res["method"])))
    for m in methods:
        i = init[init["method"] == m].copy()
        r = res[res["method"] == m].copy()
        if i.empty and r.empty:
            continue

        # Preserve original epoch numbers
        if not i.empty:
            i["epoch_original"] = i["epoch"]
        if not r.empty:
            r["epoch_original"] = r["epoch"]

        # Renumber resume epochs to continue after the initial max
        offset = int(i["epoch"].max()) if not i.empty else 0
        if not r.empty:
            r["epoch"] = r["epoch"] + offset

        frames.extend([i, r])

    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values(["method", "epoch"]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_csv, index=False)
    print(f"Saved master CSV → {out_csv}")
    return master


