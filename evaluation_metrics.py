def compute_multilabel_macro_auc(y_true, y_prob):
    """class-wise AUC를 평균낸 macro AUROC."""
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))

    if len(aucs) == 0:
        return np.nan
    return float(np.mean(aucs))


def compute_multilabel_micro_auc(y_true, y_prob):
    """multi-label micro AUROC."""
    try:
        return float(roc_auc_score(y_true, y_prob, average="micro"))
    except ValueError:
        return np.nan


def compute_per_class_auc(y_true, y_prob, class_names):
    """클래스별 AUC dict 반환."""
    out = {}
    for idx, name in enumerate(class_names):
        if len(np.unique(y_true[:, idx])) < 2:
            out[name] = np.nan
        else:
            out[name] = float(roc_auc_score(y_true[:, idx], y_prob[:, idx]))
    return out


def compute_f1_macro(y_true, y_prob, threshold=0.5):
    """threshold 기반 macro F1."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_f1_micro(y_true, y_prob, threshold=0.5):
    """threshold 기반 micro F1."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def compute_per_class_f1(y_true, y_prob, class_names, threshold=0.5):
    """클래스별 F1 dict 반환."""
    y_pred = (y_prob >= threshold).astype(int)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    out = {}
    for idx, name in enumerate(class_names):
        out[name] = float(f1s[idx])
    return out