# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/30 10:23
@file: BinaryClassifier.py
@desc: 
"""
from typing import Union, Dict

import numpy as np
from numpy import ndarray
from pandas import Series
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

__SMOOTH__ = 1e-6


def calc_ks(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], pos_label: Union[int, str] = 1) -> float:
    """
    计算ks
    :param y_true:
    :param y_pred:
    :param pos_label: positive value
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)
    threshold = thresholds[np.argmax(np.fabs(fpr - tpr))]
    ks = round(float(np.max(np.fabs(fpr - tpr))), 4)
    return ks, threshold


def calc_gini(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], average: str = 'macro') -> float:
    """
    计算gini
    :param y_true:
    :param y_pred:
    :param average: auc method
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    auc = roc_auc_score(y_true, y_pred, average=average)
    gini = round(float(2 * auc - 1), 4)
    return gini


def calc_roc(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], pos_label: Union[int, str] = 1) -> Dict:
    """
    计算roc
    :param y_true:
    :param y_pred:
    :param pos_label:
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    res = {"fpr": np.around(fpr, 4).tolist(),
           "tpr": np.around(tpr, 4).tolist()}
    return res


def calc_auc(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], average: str = 'macro') -> float:
    """
    计算auc
    :param y_true:
    :param y_pred:
    :param average: auc method
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    auc = round(float(roc_auc_score(y_true, y_pred, average=average)), 4)
    return auc


def calc_acc(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], threshold: float = None) -> float:
    """
    计算准确率
    :param y_true:
    :param y_pred:
    :param threshold:
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    if threshold is not None:
        y_pred = np.where(y_pred > threshold, 1, 0)

    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    return acc


def calc_precision(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], average: str = 'binary',
                   pos_label: Union[int, str] = 1, threshold: float = None) -> float:
    """
    计算精确度
    :param y_true:
    :param y_pred:
    :param average:
    :param pos_label:
    :param threshold:
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    if threshold is not None:
        y_pred = np.where(y_pred > threshold, 1, 0)

    precision = round(float(precision_score(y_true, y_pred, average=average, pos_label=pos_label)), 4)
    return precision


def calc_recall(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], average: str = 'binary',
                pos_label: Union[int, str] = 1, threshold: float = None) -> float:
    """
    计算召回率
    :param y_true:
    :param y_pred:
    :param average:
    :param pos_label:
    :param threshold:
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    if threshold is not None:
        y_pred = np.where(y_pred > threshold, 1, 0)

    recall = round(float(recall_score(y_true, y_pred, average=average, pos_label=pos_label)), 4)
    return recall


def calc_f1(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], average: str = 'binary',
            pos_label: Union[int, str] = 1, threshold: float = None) -> float:
    """
    计算f1值
    :param y_true:
    :param y_pred:
    :param average:
    :param pos_label:
    :param threshold:
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    if threshold is not None:
        y_pred = np.where(y_pred > threshold, 1, 0)

    f1 = round(float(f1_score(y_true, y_pred, average=average, pos_label=pos_label)), 4)
    return f1


def calc_lift(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], pos_label: Union[int, str] = 1,
              n: int = 10) -> Dict:
    """
    计算lift, 预测概率从大到小排序，选取n个截断点，计算lift值L= (TP/(TP+FP))/((TP+FP)/(TP+FP+TN+FN))
    :param y_true:
    :param y_pred:
    :param pos_label:
    :param n:
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    index = np.argsort(-y_pred)
    pred = y_pred[index]
    y_true = y_true[index]
    threshold = [int(len(pred) * (i + 1) / n) for i in range(n)]
    res = {'base': [1.0] * 10}
    lift = []

    for p in threshold:
        P = len(y_true[:p])
        TP = (y_true[:p] == pos_label).sum()
        temp = round((TP / P) / ((y_true == pos_label).sum() / y_true.size), 4)
        lift.append(temp)

    res['lift'] = lift

    return res


def calc_gain(y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series], pos_label: Union[int, str] = 1,
              n: int = 10) -> Dict:
    """
    计算gian图相关信息
    :param y_true:
    :param y_pred:
    :param pos_label:
    :param n:
    :return:
    """
    if isinstance(y_true, Series):
        y_true = y_true.values
    if isinstance(y_pred, Series):
        y_pred = y_pred.values

    index = np.argsort(-y_pred)
    y_pred = y_pred[index]
    y_true = y_true[index]
    threshold = [int(len(y_pred) * (i + 1) / n) for i in range(n)]
    res = {'base': [round(i * 0.1, 1) for i in range(n + 1)]}
    gain = [0]

    for p in threshold:
        TP = (y_true[:p] == pos_label).sum()
        temp = round(TP / (y_true == pos_label).sum(), 4)
        gain.append(temp)

    res['gain'] = gain

    return res


def calc_psi(pred1: Union[Series, ndarray], pred2: Union[Series, ndarray], min_: Union[int, float] = 300,
             max_: Union[int, float] = 1000):
    """
    计算psi
    :param pred1:
    :param pred2:
    :param min_:
    :param max_:
    :return:
    """
    if isinstance(pred1, Series):
        pred1 = pred1.values
    if isinstance(pred2, Series):
        pred2 = pred2.values

    psi_score = 0
    n = int(np.ceil((max_ - min_) / 50))
    for i in range(n):
        left_interval = int(min_ + i * 50)
        if i == (n - 1):
            right_interval = int(max_ + 1)
        else:
            right_interval = int(min_ + (i + 1) * 50)
        mask1 = (pred1 >= left_interval) & (pred1 < right_interval)
        mask2 = (pred2 >= left_interval) & (pred2 < right_interval)
        a = mask1.sum() / pred1.size
        b = mask2.sum() / pred2.size
        tmp = (a - b) * np.log((a + __SMOOTH__) / (b + __SMOOTH__))
        psi_score += tmp

    return round(float(psi_score), 6)
