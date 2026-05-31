def update_confusion(pred, target, counts):
    pred = pred.astype(bool)
    target = target.astype(bool)
    counts["tp"] += int((pred & target).sum())
    counts["fp"] += int((pred & ~target).sum())
    counts["fn"] += int((~pred & target).sum())
    counts["tn"] += int((~pred & ~target).sum())
    return counts


def metrics_from_confusion(counts):
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)

    total = tp + fp + fn + tn
    po = oa
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total + eps)
    kappa = (po - pe) / (1.0 - pe + eps)
    fp_rate = fp / (fp + tn + eps)

    return {
        "F1": float(f1),
        "IoU": float(iou),
        "OA": float(oa),
        "Precision": float(precision),
        "Recall": float(recall),
        "Kappa": float(kappa),
        "FP_Rate": float(fp_rate),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
