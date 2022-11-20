from typing import Iterable, List, Union
from pprint import pprint
from itertools import product
import numpy as np
from sklearn.metrics import confusion_matrix as cm

def confusion_matrix(pred: List[int], actual: List[int], 
                     max_class: Union[int,None] = None,
                     TP_in_topleft = False) -> List[List[int]]:
    '''
    Args:
        pred and actual are lists of the predicted class (as integers starting from 0)
        max_class: is the highest number class to include in the confusion matrix
            defaults to the max_value of the 'actual' and 'pred' lists
        TP_in_topleft: for 2 class classification, if you want class 1 (positive) 
            to be in top left instead of class 0 (negative)
    '''
    if min([*pred, *actual]) != 0:
        raise ValueError('classes should start with label int(0)')
    
    max_class = max_class if max_class is not None else max([*pred, *actual])
    cm = np.zeros((max_class+1, max_class+1))

    for p, a in zip(pred, actual):
        cm[a][p] += 1

    if TP_in_topleft:
        return np.array(
            [[cm[1][1], cm[1][0]],
            [cm[0][1], cm[0][0]]]    
        )

    return cm

def _get_cm_stats(confusion_matrix: List[List[int]]):
    '''
    Args:
        a 2D confusion matrix
    
    Returns: TP, TN, FP, FN
    '''
    TP = confusion_matrix[1][1] 
    TN = confusion_matrix[0][0] 
    FP = confusion_matrix[0][1] 
    FN = confusion_matrix[1][0]

    return TP, TN, FP, FN

def accuracy(confusion_matrix: List[List[int]]):
    '''
    If you are using a multiclass confusion matrix, please convert it to a 
    'one versus all' confusion matrix before calculating this matric

    Args:
        a 2D confusion matrix
    '''
    TP, TN, FP, FN = _get_cm_stats(confusion_matrix)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc

def error_rate(confusion_matrix: List[List[int]]):
    '''
    Calculates the error rate of a 2d classification

    If you are using a multiclass confusion matrix, please convert it to a 
    'one versus all' confusion matrix before calculating this matric

    Args:
        a 2D confusion matrix
    '''
    TP, TN, FP, FN = _get_cm_stats(confusion_matrix)
    e = (FP + FN) / (TP + TN + FP + FN)
    return e

def precision(confusion_matrix: List[List[int]]):
    '''
    Calculates the precision of a 2d classification based on the confusion marix

    If you are using a multiclass confusion matrix, please convert it to a 
    'one versus all' confusion matrix before calculating this matric

    Args:
        a 2D confusion matrix
    '''
    TP, TN, FP, FN = _get_cm_stats(confusion_matrix)
    p = TP / (TP + FP)
    return p

def recall(confusion_matrix: List[List[int]]):
    '''
    Calculates the recall of a 2d classification based on the confusion marix

    If you are using a multiclass confusion matrix, please convert it to a 
    'one versus all' confusion matrix before calculating this matric

    Args:
        a 2D confusion matrix
    '''
    TP, TN, FP, FN = _get_cm_stats(confusion_matrix)
    return TP / (TP + FN)

def f1_score(confusion_matrix: List[List[int]]):
    '''
    Calculates the f1 score of a 2d classification based on the confusion marix

    If you are using a multiclass confusion matrix, please convert it to a 
    'one versus all' confusion matrix before calculating this matric

    Args:
        a 2D confusion matrix
    '''
    P = precision(confusion_matrix)
    R = recall(confusion_matrix)
    f1 = (2 * P * R) / (P + R)
    return f1

def fpr_score(confusion_matrix: List[List[int]]):
    '''
    Calculates the False Positive Rate (Type I Error) of a 2d classification 
    based on the confusion marix

    If you are using a multiclass confusion matrix, please convert it to a 
    'one versus all' confusion matrix before calculating this matric

    Args:
        a 2D confusion matrix
    '''
    TP, TN, FP, FN = _get_cm_stats(confusion_matrix)
    fpr = FP / (FP + TN)
    return fpr

def fnr_score(confusion_matrix: List[List[int]]):
    '''
    Calculates the False Negative Rate (Type II Error) of a 2d classification 
    based on the confusion marix

    If you are using a multiclass confusion matrix, please convert it to a 
    'one versus all' confusion matrix before calculating this matric

    Args:
        a 2D confusion matrix
    '''
    TP, TN, FP, FN = _get_cm_stats(confusion_matrix)
    fnr = FN / (FN + TP)
    return fnr

def get_metrics(confusion_matrix: List[List[int]], print_metrics=True):
    '''
    Returns a dictionary with all the specified metrics

    Args:
        a 2D confusion matrix
    '''
    metrics = {
        'accuracy': accuracy(confusion_matrix),
        'precision': precision(confusion_matrix),
        'recall': recall(confusion_matrix),
        'f1': f1_score(confusion_matrix),
        'FPR': fpr_score(confusion_matrix),
        'FNR': fnr_score(confusion_matrix), 
    }

    if print_metrics:
        print('\n')
        print('\n')
        print('-' * 69)

        print('Input Confusion Matrix:')
        print(confusion_matrix)

        print('-' * 30, 'metrics', '-' * 30)
        for k, v in metrics.items():
            print(k, v, sep='\t')

        print('-' * 69)
    
    return metrics

def multiclass_accuracy(confusion_matrix: List[List[int]]):
    '''
    Function that calculates the multiclass accuracy based on a nxn sized 
    confusion matrix. It is defined as the number of correct classifications 
    divided by the total number of classifications
    '''
    n_correct = 0
    n_total = 0
    num_classes = len(confusion_matrix)

    for i, j in product(range(num_classes), range(num_classes)):
        n_total += confusion_matrix[i][j]
        if i == j:
            n_correct += confusion_matrix[i][j]

    return n_correct / n_total

def _class_confusion_matrix(confusion_matrix: List[List[int]], 
                            target_class: int):
    '''
    Takes in a nxn confusion matrix and returns a 2x2 confusion matrix for that
    specific class
    '''
    num_classes = len(confusion_matrix)
    diag = [confusion_matrix[i][i] for i in range(num_classes)]
    TP = diag[target_class] 
    TN = sum(diag) - diag[target_class]
    FP = sum(confusion_matrix[i][target_class] 
             for i in range(num_classes) if i != target_class)
    FN = sum(confusion_matrix[target_class][i] 
             for i in range(num_classes) if i != target_class) 

    class_cm = [[TP, FP],
                [FN, TN]]

    return class_cm
    
    
def macro_average_metrics(confusion_matrix: List[List[int]]):
    '''
    Function that calculates the macro_average metrics based on a nxn sized 
    confusion matrix. Macro averare is achieved by taking the metric for each 
    class, then taking the average of the class-specific-metrics
    '''
    metrics = ['precision', 'recall', 'f1']
    class_metrics = []
    num_classes = len(confusion_matrix)
    for c in range(num_classes):
        class_cm = _class_confusion_matrix(confusion_matrix, target_class=c)
        class_metrics.append(get_metrics(class_cm, print_metrics=False))
    
    macro_average_metrics = {}
    for metric in metrics:
        average_metric = sum(class_metrics[i][metric] 
                             for i in range(num_classes)) / num_classes
        macro_average_metrics[metric] = average_metric
    
    return macro_average_metrics
        



def micro_average_metrics(confusion_matrix: List[List[int]]):
    '''
    Function that calculates the macro_average metricsø precision, recall and f1 based on a nxn sized 
    confusion matrix. It is defined as the number of correct classifications 
    divided by the total number of classifications
    '''
    num_classes = len(confusion_matrix)

    tot_TP = 0
    tot_TP_FP = 0
    tot_TP_FN = 0

    for c in range(num_classes):
        class_cm = _class_confusion_matrix(confusion_matrix, c)
        TP, TN, FP, FN = _get_cm_stats(class_cm)
        tot_TP += TP
        tot_TP_FP += (TP + FP)
        tot_TP_FN += (TP + FN)

    precision_micro = tot_TP / tot_TP_FP
    recall_micro = tot_TP / tot_TP_FN
    f1_micro = (2 * precision_micro * recall_micro)\
                 / (precision_micro + recall_micro)

    micro_average_metrics = {
        'precision': precision_micro,
        'recall': recall_micro,
        'f1': f1_micro,
    }

    return micro_average_metrics


def main():
    actual =    [0, 0, 1, 0, 1, 1, 0, 1]
    pred1 =     [1, 0, 1, 1, 0, 1, 1, 1]
    pred2 =     [0, 1, 1, 0, 1, 1, 0, 0]

    cm1 = confusion_matrix(pred1, actual, TP_in_topleft=False)
    cm2 = confusion_matrix(pred2, actual, TP_in_topleft=False)

    get_metrics(cm1)
    get_metrics(cm2)



if __name__ == '__main__':
    main()