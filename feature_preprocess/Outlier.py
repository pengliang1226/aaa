# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/30 15:56
@file: Outlier.py
@desc: 异常值处理
"""
import numpy as np

from pandas import Series


def z_score_check(X: Series, threshold: float = 3):
    """
    基于3-sigma原则，检测异常值
    :param X:
    :param threshold:
    :return: 返回异常值位置
    """
    X = X.copy()

    mean = X.mean()
    std = X.std()
    score = ((X - mean) / std).abs()
    mask = score > threshold

    return mask


def IQR_check(X: Series):
    """
    根据四分位距检测异常值
    :param X:
    :return: 返回异常值位置
    """
    X = X.copy()

    a = np.nanpercentile(X, 25, axis=0)
    b = np.nanpercentile(X, 75, axis=0)

    down = a - 1.5 * (b - a)
    up = b + 1.5 * (b - a)
    mask = (X > up) | (X < down)

    return mask
