import math
import numpy as np
from sklearn import metrics
import torch

# ========= t-based 95%CI =========
_T_CRIT_975 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
               11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,24:2.064,30:2.042}

def _tcrit(n):
    df = max(n-1, 1)
    if df in _T_CRIT_975: return _T_CRIT_975[df]
    if df > max(_T_CRIT_975): return 1.96
    return _T_CRIT_975[min(_T_CRIT_975.keys(), key=lambda k: abs(k-df))]

def ci_halfwidth(std, n): 
    return _tcrit(n) * (std / math.sqrt(n))

def _ensure_binary_labels(y):
    y = np.asarray(y).astype(int)
    uniq = np.unique(y)
    if set(uniq.tolist()) != {0,1}:
        raise ValueError(f"Labels must be binary {{0,1}}, got {uniq}")
    return y

@torch.no_grad()
def collect_probs_labels(model, data_loader, device, force_g_zero=False):
    model.eval()
    probs, labels = [], []
    for trad, esm, y in data_loader:
        trad, esm = trad.to(device), esm.to(device)
        y = y.to(device)
        logits, _, _ = model(trad, esm, force_g_zero=force_g_zero)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(p.detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())
    return np.array(probs), np.array(labels)

def eval_with_threshold(probs, labels, thr):
    labels = _ensure_binary_labels(labels)
    preds = (np.asarray(probs) > thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds, labels=[0,1]).ravel()
    acc = 100.0 * (tp + tn) / (tp + tn + fp + fn + 1e-12)
    sen = tp / (tp + fn + 1e-12)
    spe = tn / (tn + fp + 1e-12)
    f1  = metrics.f1_score(labels, preds)
    mcc = metrics.matthews_corrcoef(labels, preds)
    auc = metrics.roc_auc_score(labels, probs)
    return acc, sen, spe, f1, mcc, auc
