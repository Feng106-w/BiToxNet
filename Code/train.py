import os
import time
import copy
import json
import math
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import dvib_ban
from evaluate import (
    collect_probs_labels, eval_with_threshold, ci_halfwidth
)


SEED = 42
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)


def load_trad_full(csv_path):
    df = pd.read_csv(csv_path)
    if 'Label' not in df.columns:
        raise ValueError("Input CSV must contain 'Label' column.")
    drop_cols = [c for c in ['Label','Id','Sequence'] if c in df.columns]
    feature_names = [c for c in df.columns if c not in drop_cols]
    X = df[feature_names].values
    y = df['Label'].values.astype(int)
    return X, y, feature_names


def load_esm(csv_path):
    df = pd.read_csv(csv_path, header=0)
    return df.values


def train_one_fold(model, train_loader, val_loader, device):
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
    best_auc, best_state, best_ep, noimp = -1.0, None, 0, 0
    for ep in range(1, EPOCHS+1):
        model.train()
        ep_gate_means = []
        for trad, esm, y in train_loader:
            trad, esm, y = trad.to(device), esm.to(device), y.to(device)
            optim.zero_grad()
            force_g_zero = (ep <= WARMUP_Q_ONLY_EPOCHS)
            logits, _, g_mean = model(trad, esm, force_g_zero=force_g_zero)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()
            ep_gate_means.append(g_mean)
        sched.step()
        vprob, vlab = collect_probs_labels(model, val_loader, device, force_g_zero=False)
        try:
            vauc = metrics.roc_auc_score(vlab, vprob)
        except ValueError:
            vauc = float('nan')
        print(f"[Epoch {ep:03d}] ValAUC={vauc:.4f} | gate_mean={np.mean(ep_gate_means):.4f}")
        if not np.isnan(vauc) and vauc > best_auc:
            best_auc, best_state, best_ep, noimp = vauc, copy.deepcopy(model.state_dict()), ep, 0
        else:
            noimp += 1
            if noimp >= PATIENCE:
                print(f"Early stop at {ep} (no improve {PATIENCE}).")
                break
    return best_state, best_ep, best_auc


def main():
    X_trad_train, y_train, feat_names = load_trad_full(TRAIN_TRAD_CSV)
    X_trad_test,  y_test,  _         = load_trad_full(TEST_TRAD_CSV)
    
    X_esm_train = load_esm(TRAIN_ESM)
    X_esm_test  = load_esm(TEST_ESM)

    assert X_trad_train.shape[0] == X_esm_train.shape[0] == len(y_train)
    assert X_trad_test.shape[0]  == X_esm_test.shape[0]  == len(y_test)

    esm_dim  = X_esm_train.shape[1]
    print(f"trad_input_dim(full)={X_trad_train.shape[1]}, esm_input_dim={esm_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(len(y_train), dtype=float)
    per_fold_rows = []

    global_best = {
        "fold": None,
        "val_auc_earlystop": -1.0,
        "state_dict": None,
        "trad_dim": None,
        "selector_path": None
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_trad_train, y_train), start=1):
        t0 = time.time()
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

        X_tr_trad, y_tr = X_trad_train[tr_idx], y_train[tr_idx]
        X_va_trad, y_va = X_trad_train[val_idx], y_train[val_idx]
        X_tr_esm,  X_va_esm = X_esm_train[tr_idx], X_esm_train[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_trad)

        k = min(PRESELECT_K, X_tr_scaled.shape[1])
        pre_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        pre_selector.fit(X_tr_scaled, y_tr)

        X_tr_pre = pre_selector.transform(X_tr_scaled)

        base_est = LogisticRegression(max_iter, penalty, solver, random_state=SEED)
        rfecv = RFECV(estimator=base_est, step=RFE_STEP, cv=INNER_CV,
                      scoring='roc_auc', n_jobs=-1, min_features_to_select)
        rfecv.fit(X_tr_pre, y_tr)

    
        pre_mask = pre_selector.get_support()
        pre_names = [n for n, m in zip(feat_names, pre_mask) if m]
        rfe_mask_in_pre = rfecv.support_.astype(bool)
        final_names = [n for n, m in zip(pre_names, rfe_mask_in_pre) if m]
        pd.Series(final_names).to_csv(os.path.join(OUT_DIR, f" "), index=False)

    
        X_va_scaled = scaler.transform(X_va_trad)
        X_tr_sel = rfecv.transform(X_tr_pre)
        X_va_sel = rfecv.transform(pre_selector.transform(X_va_scaled))
        trad_dim_fold = X_tr_sel.shape[1]
        print(f"[Fold {fold}] Preselect {k} -> RFECV selected {trad_dim_fold} features.")

    
        tr_ds = TensorDataset(torch.FloatTensor(X_tr_sel), torch.FloatTensor(X_tr_esm), torch.LongTensor(y_tr))
        va_ds = TensorDataset(torch.FloatTensor(X_va_sel), torch.FloatTensor(X_va_esm), torch.LongTensor(y_va))
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    
        model = dvib_ban(
            trad_input_dim=trad_dim_fold, esm_input_dim=esm_dim,
            reduce_dim=REDUCE_DIM, hidden_size=HIDDEN_SIZE,
            dropout=DROPOUT, ban_heads=BAN_HEADS, ban_dropout=BAN_DROPOUT
        ).to(device)
        best_state, best_ep, best_auc = train_one_fold(model, tr_loader, va_loader, device)
        model.load_state_dict(best_state)

    
        va_probs, _ = collect_probs_labels(model, va_loader, device, force_g_zero=False)
        oof_probs[val_idx] = va_probs

    
        acc, sen, spe, f1, mcc, auc_at_thr = eval_with_threshold(va_probs, y_va, FIXED_THRESHOLD)
        print(
            f"[Fold {fold}] VAL (thr={FIXED_THRESHOLD:.2f}) "
            f"Acc={acc:.2f}% | Sen={sen:.4f} | Spe={spe:.4f} | MCC={mcc:.4f} | "
            f"AUC={metrics.roc_auc_score(y_va, va_probs):.4f}"
        )
        
        per_fold_rows.append({
            "fold": fold,
            "preselect_k": int(k),
            "selected_features": trad_dim_fold,
            "best_epoch": best_ep,
            "val_auc_earlystop": float(best_auc),
            "thr_used": float(FIXED_THRESHOLD),
            "val_acc": float(acc),
            "val_sen": float(sen),
            "val_spe": float(spe),
            "val_f1":  float(f1),
            "val_mcc": float(mcc),
            "val_auc": float(metrics.roc_auc_score(y_va, va_probs)),
            "train_time_sec": time.time() - t0
        })

    
        sel_path = os.path.join(OUT_DIR, f" ")
        joblib.dump(
            {"scaler": scaler, "pre_selector": pre_selector, "rfecv": rfecv, "feature_names": feat_names},
            sel_path
        )

    
        if best_auc > global_best["val_auc_earlystop"]:
            global_best.update({
                "fold": fold,
                "val_auc_earlystop": float(best_auc),
                "state_dict": copy.deepcopy(best_state),
                "trad_dim": int(trad_dim_fold),
                "selector_path": sel_path
            })


    pd.DataFrame({
        "index": np.arange(len(y_train)),
        "oof_prob": oof_probs,
        "label": y_train
    }).to_csv(os.path.join(OUT_DIR, " "), index=False)

    oof_acc, oof_sen, oof_spe, oof_f1, oof_mcc, oof_auc = eval_with_threshold(oof_probs, y_train, FIXED_THRESHOLD)


    df_fold = pd.DataFrame(per_fold_rows)
    df_fold.to_csv(os.path.join(OUT_DIR, " "), index=False)

    def _ms(a): return float(np.mean(a)), float(np.std(a))
    n = len(df_fold)
    v_acc_m, v_acc_s = _ms(df_fold["val_acc"])
    v_sen_m, v_sen_s = _ms(df_fold["val_sen"])
    v_spe_m, v_spe_s = _ms(df_fold["val_spe"])
    v_f1_m,  v_f1_s  = _ms(df_fold["val_f1"])
    v_mcc_m, v_mcc_s = _ms(df_fold["val_mcc"])
    v_auc_m, v_auc_s = _ms(df_fold["val_auc"])
    tt_m,    tt_s    = _ms(df_fold["train_time_sec"])

    summary_oof = {
        "folds": n,
        "threshold_used": float(FIXED_THRESHOLD),
        "oof_global": {
            "acc": oof_acc, "sen": oof_sen, "spe": oof_spe,
            "f1": oof_f1, "mcc": oof_mcc, "auc": oof_auc
        },
        "per_fold_stats": {
            "acc":  {"mean": v_acc_m, "std": v_acc_s, "ci95_halfwidth": ci_halfwidth(v_acc_s, n)},
            "sen":  {"mean": v_sen_m, "std": v_sen_s, "ci95_halfwidth": ci_halfwidth(v_sen_s, n)},
            "spe":  {"mean": v_spe_m, "std": v_spe_s, "ci95_halfwidth": ci_halfwidth(v_spe_s, n)},
            "f1":   {"mean": v_f1_m,  "std": v_f1_s,  "ci95_halfwidth": ci_halfwidth(v_f1_s, n)},
            "mcc":  {"mean": v_mcc_m, "std": v_mcc_s, "ci95_halfwidth": ci_halfwidth(v_mcc_s, n)},
            "auc":  {"mean": v_auc_m, "std": v_auc_s, "ci95_halfwidth": ci_halfwidth(v_auc_s, n)},
            "train_time_sec": {"mean": tt_m, "std": tt_s, "ci95_halfwidth": ci_halfwidth(tt_s, n)}
        },
        "selected_best_fold": {
            "fold": global_best["fold"],
            "val_auc_earlystop": global_best["val_auc_earlystop"]
        }
    }
    with open(os.path.join(OUT_DIR, " "), "w") as f:
        json.dump(summary_oof, f, indent=2)

    print("\n===== OOF SUMMARY (Fixed threshold) =====")
    print(f"Thr={FIXED_THRESHOLD:.3f} | AUC={oof_auc:.4f}")
    print(f"Acc={oof_acc:.2f}% | F1={oof_f1:.4f} | MCC={oof_mcc:.4f} | SEN={oof_sen:.4f} | SPE={oof_spe:.4f}")
    print(f"Selected BEST FOLD = {global_best['fold']} | ValAUC(earlystop) = {global_best['val_auc_earlystop']:.4f}")

    
    best_sel = joblib.load(global_best["selector_path"])
    scaler_full = best_sel["scaler"]
    pre_selector_full = best_sel["pre_selector"]
    rfecv_full = best_sel["rfecv"]

    trad_dim_final = global_best["trad_dim"]
    model_final = dvib_ban(
        trad_input_dim=trad_dim_final, esm_input_dim=esm_dim,
        reduce_dim=REDUCE_DIM, hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT, ban_heads=BAN_HEADS, ban_dropout=BAN_DROPOUT
    ).to(device)
    model_final.load_state_dict(global_best["state_dict"])

    X_test_scaled = scaler_full.transform(X_trad_test)
    X_test_pre = pre_selector_full.transform(X_test_scaled)
    X_test_sel = rfecv_full.transform(X_test_pre)

    te_ds = TensorDataset(torch.FloatTensor(X_test_sel), torch.FloatTensor(X_esm_test), torch.LongTensor(y_test))
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    test_probs, test_labels = collect_probs_labels(model_final, te_loader, device, force_g_zero=False)
    pd.DataFrame({"test_prob": test_probs, "label": test_labels}).to_csv(
        os.path.join(OUT_DIR, " "), index=False
    )
    test_acc, test_sen, test_spe, test_f1, test_mcc, test_auc = eval_with_threshold(test_probs, test_labels, FIXED_THRESHOLD)

    test_summary = {
        "used_fold": int(global_best["fold"]),
        "val_auc_earlystop_on_used_fold": float(global_best["val_auc_earlystop"]),
        "threshold_used": float(FIXED_THRESHOLD),
        "acc": float(test_acc), "sen": float(test_sen), "spe": float(test_spe),
        "f1": float(test_f1), "mcc": float(test_mcc), "auc": float(test_auc)
    }
    with open(os.path.join(OUT_DIR, " "), "w") as f:
        json.dump(test_summary, f, indent=2)

    torch.save({
        "model_state_dict": model_final.state_dict(),
        "trad_input_dim": trad_dim_final,
        "esm_input_dim": esm_dim,
        "best_fold": int(global_best["fold"]),
        "val_auc_best_fold": float(global_best["val_auc_earlystop"]),
        "threshold_used": float(FIXED_THRESHOLD)
    }, os.path.join(OUT_DIR, " "))

    print("\n===== FINAL TEST (Fixed threshold; single evaluation with the best-fold pipeline) =====")
    print(f"BestFold={global_best['fold']} | Thr={FIXED_THRESHOLD:.3f} | "
          f"AUC={test_auc:.4f} | Acc={test_acc:.2f}% | F1={test_f1:.4f} | "
          f"MCC={test_mcc:.4f} | SEN={test_sen:.4f} | SPE={test_spe:.4f}")
    print(f"All artifacts saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
