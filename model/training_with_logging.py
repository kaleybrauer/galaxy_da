import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from geomloss import SamplesLoss

from metrics_alignment import compute_epoch_metrics

# ---- Gradient Reversal Layer for DANN ----
class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GradReverse(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradReverseFn.apply(x, self.lambd)

# ---- Domain Discriminator (simple MLP on features) ----
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=128, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2)  # 2 domains: source(0), target(1)
        )
    def forward(self, z): return self.net(z)

# ---- quick eval acc ----
@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    tot=0; correct=0
    for x,y in loader:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        logits,_ = model(x)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        tot += y.size(0)
    return 100.0 * correct / max(1, tot)

# ---- main training loop with logging ----
def train_with_lambda_logging(
    lambda_val,
    method_name,                      # "NoAdapt" | "MMD" | "Sinkhorn" | "DANN"
    model_ctor,                       # e.g. lambda: CNN(num_classes=3).to(device)
    optimizer_ctor,                   # e.g. lambda params: torch.optim.AdamW(params, lr=1e-3)
    train_loader_source,
    train_loader_target,
    eval_loader_source,
    eval_loader_target,
    device="cuda",
    num_epochs=5,
    blur=0.05,                        # for Sinkhorn geomloss/divergence
    class_names=("elliptical","irregular","spiral"),
    use_true_target_labels=True       # True for evaluation; training still unsupervised on target
):
    torch.cuda.empty_cache()
    model = model_ctor().to(device)

    try:
        ell_idx = [i for i, n in enumerate(class_names) if str(n).lower() == "elliptical"][0]
    except IndexError:
        ell_idx = 0  # fallback if name not found
    class_weights = torch.ones(len(class_names), dtype=torch.float32, device=device)
    class_weights[ell_idx] = 3.0  # weight ellipticals 3x

    # DANN needs a discriminator + GRL
    disc = None; grl = None
    if method_name.lower()=="dann":
        in_dim = getattr(model, "feature_dim", 128)
        disc = DomainDiscriminator(in_dim=in_dim, hidden=128).to(device)
        grl = GradReverse(lambd=lambda_val)  # lambda_val acts as adv weight
        params = list(model.parameters()) + list(disc.parameters())
        opt = optimizer_ctor(params)
        dom_criterion = nn.CrossEntropyLoss()  # domain labels (unweighted)
    else:
        opt = optimizer_ctor(model.parameters())

    ce = nn.CrossEntropyLoss(weight=class_weights)

    aux_loss_fn = None
    if method_name.lower()=="mmd":
        aux_loss_fn = SamplesLoss("energy", p=2)     # energy distance (fast, stable)
    if method_name.lower()=="sinkhorn":
        aux_loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur, debias=True)

    rows = []

    for epoch in range(1, num_epochs+1):
        model.train()
        if disc is not None: disc.train()

        run_tot = run_ce = run_aux = 0.0
        steps = 0
        tr_src_correct=tr_tgt_correct=0
        tr_src_tot=tr_tgt_tot=0

        iterator = zip(train_loader_source, train_loader_target)
        nsteps = min(len(train_loader_source), len(train_loader_target))

        for (s_x, s_y), (t_x, t_y) in tqdm(iterator, total=nsteps, desc=f"{method_name} λ={lambda_val} epoch {epoch}/{num_epochs}", leave=False):
            s_x = s_x.to(device, non_blocking=True); s_y = s_y.to(device, non_blocking=True)
            t_x = t_x.to(device, non_blocking=True)

            bs = min(s_x.size(0), t_x.size(0))
            s_x, s_y = s_x[:bs], s_y[:bs]
            t_x       = t_x[:bs]

            if method_name.lower()=="dann":
                # forward source
                logits_s, z_s = model(s_x)
                # weighted CE on source labels (ellipticals 3x)
                ce_loss = ce(logits_s, s_y)

                # forward target (for domain only)
                _, z_t = model(t_x)

                # domain loss with GRL (unweighted)
                z_mix = torch.cat([z_s, z_t], 0)
                z_mix = F.normalize(z_mix, dim=1)
                z_mix = grl(z_mix)  # gradients reversed
                dom_logits = disc(z_mix)
                dom_labels = torch.cat([torch.zeros(bs, dtype=torch.long, device=device),
                                        torch.ones(bs,  dtype=torch.long, device=device)], 0)
                aux_loss = dom_criterion(dom_logits, dom_labels)

                total_loss = ce_loss + aux_loss  # GRL scales gradients; optionally: ce_loss + lambda_val*aux_loss
            else:
                # joint forward for MMD/Sinkhorn/NoAdapt
                x = torch.cat([s_x, t_x], 0)
                logits, z = model(x)
                logits_s, logits_t = logits[:bs], logits[bs:]
                z_s, z_t = z[:bs], z[bs:]
                z_s = F.normalize(z_s, dim=1); z_t = F.normalize(z_t, dim=1)

                # weighted CE on source labels (ellipticals 3x)
                ce_loss = ce(logits_s, s_y)
                if method_name.lower() in ("mmd","sinkhorn"):
                    aux_loss = aux_loss_fn(z_s, z_t)
                    total_loss = ce_loss + lambda_val * aux_loss
                else:  # NoAdapt
                    aux_loss = torch.tensor(0.0, device=device)
                    total_loss = ce_loss

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()

            # stats
            with torch.no_grad():
                run_tot += float(total_loss.item())
                run_ce  += float(ce_loss.item())
                run_aux += float(aux_loss.item()) if torch.is_tensor(aux_loss) else float(aux_loss)
                steps += 1

                # quick train acc readbacks
                if method_name.lower()=="dann":
                    tr_src_correct += (logits_s.argmax(1)==s_y).sum().item()
                    tr_src_tot     += bs
                else:
                    tr_src_correct += (logits_s.argmax(1)==s_y).sum().item()
                    tr_src_tot     += bs
                    tr_tgt_correct += 0  # not meaningful; keep 0

        # epoch-end evals
        train_src_acc = 100.0 * tr_src_correct / max(1, tr_src_tot)
        eval_src_acc = eval_accuracy(model, eval_loader_source, device)
        eval_tgt_acc = eval_accuracy(model, eval_loader_target, device)

        diag = compute_epoch_metrics(
            model,
            eval_loader_source=eval_loader_source,
            eval_loader_target=eval_loader_target,
            device=device,
            class_names=class_names,
            blur=blur,
            ot_reg=0.05,
            use_true_target_labels=use_true_target_labels
        )

        row = dict(
            method=method_name,
            lambda_val=lambda_val,
            epoch=epoch,
            train_total_loss=run_tot/max(1,steps),
            train_ce_loss=run_ce/max(1,steps),
            train_aux_loss=run_aux/max(1,steps),
            train_src_acc=train_src_acc,
            eval_src_acc=eval_src_acc,
            eval_tgt_acc=eval_tgt_acc,
            target_acc=diag["target_acc"],
            target_macro_f1=diag["target_macro_f1"],
            mmd2=diag["mmd2"],
            sinkhorn_div=diag["sinkhorn_div"],
            domain_auc=diag["domain_auc"],
            domain_acc=diag["domain_acc"],
            ot_on_diag=(diag["ot_on_diag"] if diag["ot_on_diag"] is not None else np.nan),
        )
        # per-class recalls & cmmd & domain AUC/class
        for cname, val in (diag["recall_per_class"] or {}).items():
            row[f"recall_{cname}"] = val
        for cname, val in (diag["cmmd"] or {}).items():
            row[f"cmmd_{cname}"] = val
        for ic, val in (diag["domain_auc_per_class"] or {}).items():
            cname = class_names[int(ic)]
            row[f"domAUC_{cname}"] = val

        rows.append(row)

    return pd.DataFrame(rows), model
