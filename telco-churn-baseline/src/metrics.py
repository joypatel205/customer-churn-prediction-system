from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def compute_metrics(y_true, y_prob):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
