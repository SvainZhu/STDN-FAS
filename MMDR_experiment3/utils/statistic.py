import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def calculate_statistic(probs, labels):
    TN, FP, FN, TP = confusion_matrix(labels, probs).ravel()
    APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
    NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
    ACER = (APCER + NPCER) / 2.0
    ACC = (TP + TN) / len(labels)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return APCER, NPCER, ACER, ACC, HTER

def calculate_accuracy_score(labels, scores):
    return accuracy_score(labels, scores)

def calculate_roc_auc_score(labels, scores):
    return roc_auc_score(labels, scores)