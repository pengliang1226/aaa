# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/30 15:47
@file: Imputer.py
@desc: 缺失值填充
"""
from typing import List

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.impute import KNNImputer


def statistics_imputer(X: Series, method: str = 'mean', null_value: List = None) -> Series:
    """
    统计指标填充，例如：均值、中位数、众数等
    :param X:
    :param method: 目前仅支持均值、中位数、众数、最大值、最小值
    :param null_value: 缺失值列表
    :return:
    """
    X = X.copy()
    if null_value is not None:
        X[X.isin(null_value)] = np.nan

    if method == 'mean':
        fill_value = X.mean()
    elif method == 'median':
        fill_value = X.median()
    elif method == 'mode':
        fill_value = X.mode()[0]
    elif method == 'max':
        fill_value = X.max()
    elif method == 'min':
        fill_value = X.min()
    else:
        raise Exception('未配置的填充方法')

    X.fillna(fill_value, inplace=True)

    return X


def interpolate_imputer(X: Series, null_value: List = None, **kwargs) -> Series:
    """
    插值法填充缺失值
    :param X:
    :param null_value:
    :return:
    """
    X = X.copy()
    if null_value is not None:
        X[X.isin(null_value)] = np.nan

    X.interpolate(kwargs, inplace=True)

    return X


def knn_imputer(X: DataFrame, n_neighbors: int = 5) -> DataFrame:
    """
    k临近法填充缺失值
    :param X:
    :param n_neighbors:
    :return:
    """
    X = X.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X

# def iterative_imputer(X: DataFrame, max_iter: int = 10) -> DataFrame:
#     """
#     学习器预测填充缺失值，默认贝叶斯
#     :param X:
#     :param max_iter:
#     :return:
#     """
#     X = X.copy()
#     imputer = IterativeImputer(max_iter=max_iter, random_state=0)
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#     return X
