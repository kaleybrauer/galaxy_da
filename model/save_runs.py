from pathlib import Path
import time, hashlib
import torch, pandas as pd

def make_run_id(method_name, lambda_val):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sig = f"{method_name}|{lambda_val}|{stamp}"
    h = hashlib.sha1(sig.encode()).hexdigest()[:6]
    return f"{method_name}_lam{lambda_val}_{stamp}_{h}"

def save_end_of_run(df_metrics, model, method_name, lambda_val,
                    logs_dir="logs", ckpt_dir="ckpts", prefix=None):
    """
    Save once per method after training finishes 
    - df_metrics: DataFrame returned by your training function (per-epoch rows)
    - model: final trained model
    """
    run_id = make_run_id(method_name, lambda_val)
    name   = f"{prefix+'_' if prefix else ''}{method_name}"

    # paths
    logs_p  = Path(logs_dir); logs_p.mkdir(parents=True, exist_ok=True)
    ckpt_p  = Path(ckpt_dir); ckpt_p.mkdir(parents=True, exist_ok=True)
    csv_fp  = logs_p / f"{name}_metrics_{run_id}.csv"
    parq_fp = logs_p / f"{name}_metrics_{run_id}.parquet"
    ckpt_fp = ckpt_p / f"{name}_final_{run_id}.pt"

    # write
    df_metrics.to_csv(csv_fp, index=False)
    try:
        df_metrics.to_parquet(parq_fp, index=False)  # handy for fast reloads (optional)
    except Exception:
        pass
    torch.save(model.state_dict(), ckpt_fp)

    print(f"[{method_name}] saved metrics → {csv_fp}")
    print(f"[{method_name}] saved checkpoint → {ckpt_fp}")
    return dict(csv=str(csv_fp), parquet=str(parq_fp), ckpt=str(ckpt_fp), run_id=run_id)
