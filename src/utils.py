import pandas as pd
from sklearn.metrics import f1_score

# 8 BFRB gestures (taken from the competition overview page)
TARGET_GESTURES = [
    "Above ear - pull hair", "Forehead - pull hairline", "Forehead - scratch",
    "Eyebrow - pull hair", "Eyelash - pull hair", "Neck - pinch skin",
    "Neck - scratch", "Cheek - pinch skin",
]


def compute_hierarchical_f1(y_true, y_pred):
    # Kaggle score = (binary F1 + macro F1) / 2
    y_true = pd.Series(list(y_true))
    y_pred = pd.Series(list(y_pred))

    # part 1: just check if it's a BFRB gesture or not
    true_is_target = y_true.isin(TARGET_GESTURES)
    pred_is_target = y_pred.isin(TARGET_GESTURES)
    binary_f1 = f1_score(true_is_target, pred_is_target)

    # part 2: all non-BFRB gestures count as one class -> 9 classes
    true_9 = y_true.where(true_is_target, "non_target")
    pred_9 = y_pred.where(pred_is_target, "non_target")
    macro_f1 = f1_score(true_9, pred_9, average="macro")

    return (binary_f1 + macro_f1) / 2