import numpy as np, pandas as pd, torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    import umap  # optional; otherwise we fallback to TSNE
    HAS_UMAP = True
except Exception:
    from sklearn.manifold import TSNE
    HAS_UMAP = False

from geomloss import SamplesLoss  # for Sinkhorn divergence diagnostics (fast)

# ---------- feature extraction ----------
@torch.no_grad()
def extract_features(model, loader, device, max_batches=None, feature_normalize=True):
    """Return dict: z, y, logits, preds (numpy). Uses your CNN: model(x) -> (logits, z)."""
    model.eval()
    Z, Y, LOGITS, PREDS = [], [], [], []
    it = iter(loader)
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    for _ in range(n_batches):
        images, labels = next(it)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, z = model(images)
        if feature_normalize:
            z = F.normalize(z, dim=1)
        Z.append(z.detach().cpu().numpy())
        LOGITS.append(logits.detach().cpu().numpy())
        Y.append(labels.detach().cpu().numpy())
        PREDS.append(logits.argmax(1).detach().cpu().numpy())
    Z = np.concatenate(Z, 0) if Z else np.empty((0, 1))
    Y = np.concatenate(Y, 0) if Y else np.empty((0,), dtype=int)
    LOGITS = np.concatenate(LOGITS, 0) if LOGITS else np.empty((0, 1))
    PREDS = np.concatenate(PREDS, 0) if PREDS else np.empty((0,), dtype=int)
    return dict(z=Z, y=Y, logits=LOGITS, preds=PREDS)

# ---------- MMD^2 (Gaussian, unbiased) ----------
def _median_sigma(X, Y):
    if X.size == 0 or Y.size == 0: return 1.0
    xs = X[np.random.choice(len(X), min(512, len(X)), replace=False)]
    ys = Y[np.random.choice(len(Y), min(512, len(Y)), replace=False)]
    Dxx = ((xs[:,None]-xs[None,:])**2).sum(-1)
    Dyy = ((ys[:,None]-ys[None,:])**2).sum(-1)
    Dxy = ((xs[:,None]-ys[None,:])**2).sum(-1)
    d = np.concatenate([Dxx[np.triu_indices_from(Dxx,1)], Dyy[np.triu_indices_from(Dyy,1)], Dxy.ravel()])
    d = d[d>0]
    if d.size==0: return 1.0
    return float(np.sqrt(0.5*np.median(d)))

def _gauss(X, Y, sigma):
    D = ((X[:,None]-Y[None,:])**2).sum(-1) / (2.0*sigma**2 + 1e-12)
    return np.exp(-D)

def mmd2_unbiased_gaussian(X, Y, sigma=None):
    nx, ny = len(X), len(Y)
    if nx < 2 or ny < 2: return np.nan
    sigma = _median_sigma(X, Y) if sigma is None else sigma
    Kxx = _gauss(X, X, sigma); np.fill_diagonal(Kxx, 0.0)
    Kyy = _gauss(Y, Y, sigma); np.fill_diagonal(Kyy, 0.0)
    Kxy = _gauss(X, Y, sigma)
    term_x = Kxx.sum() / (nx*(nx-1))
    term_y = Kyy.sum() / (ny*(ny-1))
    term_xy = 2.0*Kxy.mean()
    return float(term_x + term_y - term_xy)

# ---------- Sinkhorn divergence (geomloss) ----------
def sinkhorn_divergence(X, Y, blur=0.05, p=2):
    if X.size==0 or Y.size==0: return np.nan
    loss = SamplesLoss("sinkhorn", p=p, blur=blur, debias=True)
    x = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(Y, dtype=torch.float32)
    return float(loss(x, y).item())

# ---------- Optional: OT plan → class–class mass (needs POT) ----------
def sinkhorn_plan_class_mass(X, Y, y_src, y_tgt_pred, reg=0.05, n_classes=3):
    try:
        import ot
    except Exception:
        return None, None
    ns, nt = len(X), len(Y)
    if ns==0 or nt==0: return None, None
    a = np.ones(ns)/ns; b=np.ones(nt)/nt
    M = ((X[:,None]-Y[None,:])**2).sum(-1)  # squared Euclidean
    P = ot.sinkhorn(a, b, M, reg=reg)       # (ns, nt)
    mass = np.zeros((n_classes, n_classes), dtype=np.float64)
    for cs in range(n_classes):
        sm = (y_src==cs)
        if not sm.any(): continue
        for ct in range(n_classes):
            tm = (y_tgt_pred==ct)
            if not tm.any(): continue
            mass[cs, ct] = P[np.ix_(sm, tm)].sum()
    s = mass.sum()
    if s>0: mass /= s
    on_diag = float(np.trace(mass)) if s>0 else np.nan
    return mass, on_diag

# ---------- Domain separability probes ----------
def domain_probe_auc(Xs, Xt, seed=123, max_n=4000):
    rng = np.random.default_rng(seed)
    if len(Xs)>max_n: Xs = Xs[rng.choice(len(Xs), max_n, replace=False)]
    if len(Xt)>max_n: Xt = Xt[rng.choice(len(Xt), max_n, replace=False)]
    X = np.vstack([Xs, Xt]); y = np.hstack([np.zeros(len(Xs), int), np.ones(len(Xt), int)])
    idx = rng.permutation(len(X)); ntr = int(0.7*len(X))
    tr, te = idx[:ntr], idx[ntr:]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[te])[:,1]
    preds = (probs>=0.5).astype(int)
    auc = roc_auc_score(y[te], probs); acc = (preds==y[te]).mean()
    return float(auc), float(acc)

def domain_probe_auc_per_class(Xs, ys, Xt, yt_pred, classes):
    out = {}
    for c in classes:
        Xsc = Xs[ys==c]; Xtc = Xt[yt_pred==c]
        if len(Xsc)<5 or len(Xtc)<5:
            out[int(c)] = np.nan
        else:
            out[int(c)] = domain_probe_auc(Xsc, Xtc)[0]
    return out

# ---------- One-shot metrics for an epoch ----------
def compute_epoch_metrics(
    model,
    eval_loader_source,
    eval_loader_target,
    device="cuda",
    class_names=("elliptical","irregular","spiral"),
    max_batches=None,
    blur=0.05,
    ot_reg=0.05,
    use_true_target_labels=True
):
    # features
    src = extract_features(model, eval_loader_source, device, max_batches=max_batches, feature_normalize=True)
    tgt = extract_features(model, eval_loader_target, device, max_batches=max_batches, feature_normalize=True)

    # performance on target
    y_true_t = tgt["y"]
    y_pred_t = tgt["preds"]
    acc = accuracy_score(y_true_t, y_pred_t) if y_true_t.size else np.nan
    macro_f1 = f1_score(y_true_t, y_pred_t, average="macro", zero_division=0) if y_true_t.size else np.nan
    recalls = recall_score(y_true_t, y_pred_t, average=None, labels=list(range(len(class_names))), zero_division=0) if y_true_t.size else np.array([np.nan]*len(class_names))
    cm = confusion_matrix(y_true_t, y_pred_t, labels=list(range(len(class_names)))) if y_true_t.size else np.zeros((len(class_names), len(class_names)), int)

    # alignment
    mmd2 = mmd2_unbiased_gaussian(src["z"], tgt["z"])
    sink = sinkhorn_divergence(src["z"], tgt["z"], blur=blur, p=2)

    # domain probes
    dom_auc, dom_acc = domain_probe_auc(src["z"], tgt["z"])
    y_t_for_cc = y_true_t if (use_true_target_labels and y_true_t.size) else y_pred_t

    # class-conditional MMD^2
    cmmd = {}
    for i, cname in enumerate(class_names):
        Xs = src["z"][src["y"]==i]; Xt = tgt["z"][y_t_for_cc==i]
        cmmd[cname] = mmd2_unbiased_gaussian(Xs, Xt) if (len(Xs)>1 and len(Xt)>1) else np.nan

    # classwise domain AUC (for DANN panel)
    dom_auc_cls = domain_probe_auc_per_class(src["z"], src["y"], tgt["z"], y_t_for_cc, classes=range(len(class_names)))

    # optional OT class–class mass (if POT installed)
    ot_mass, ot_on_diag = sinkhorn_plan_class_mass(src["z"], tgt["z"], src["y"], y_t_for_cc, reg=ot_reg, n_classes=len(class_names))

    # low-dim embedding (use sparingly)
    embed = None
    try:
        Z = np.vstack([src["z"], tgt["z"]])
        dom = np.hstack([np.zeros(len(src["z"]), int), np.ones(len(tgt["z"]), int)])
        lab = np.hstack([src["y"], y_t_for_cc])
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
            XY = reducer.fit_transform(Z)
        else:
            XY = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto", random_state=42).fit_transform(Z)
        embed = dict(xy=XY, domain=dom, label=lab)
    except Exception:
        embed = None

    return dict(
        target_acc=acc*100 if acc==acc else np.nan,
        target_macro_f1=macro_f1,
        recall_per_class=dict(zip(class_names, recalls.tolist())),
        confusion_matrix=cm,
        mmd2=mmd2,
        sinkhorn_div=sink,
        domain_auc=dom_auc,
        domain_acc=dom_acc,
        cmmd=cmmd,
        domain_auc_per_class=dom_auc_cls,
        ot_mass=ot_mass,
        ot_on_diag=ot_on_diag,
        embed=embed
    )
