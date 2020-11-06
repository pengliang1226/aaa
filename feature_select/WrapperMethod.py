# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/26 11:44
@file: WrapperMethod.py
@desc: 包装法，根据目标函数（学习器的性能，往往是预测效果评分），每次选择若干特征，或者排除若干特征
"""
from typing import List, Any

import numpy as np
from pandas import DataFrame, Series
from sklearn.feature_selection import RFE, RFECV


def RFE_filter(df: DataFrame, y: Series, col_list: List, estimator: Any, keep: float = 0.5, step: int = 1) -> List:
    """
    递归特征消除
    :param df:
    :param y:
    :param col_list:
    :param estimator: 使用的学习器
    :param keep: 保留特征数目或比例
    :param step: 每次递归的步长
    :return:
    """
    if keep >= 1 and isinstance(keep, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(keep, float):
        keep = np.ceil(len(col_list) * keep)

    selector = RFE(estimator, n_features_to_select=keep, step=step)
    selector = selector.fit(df[col_list], y)
    mask = selector.get_support()

    res = np.array(col_list)[mask].tolist()

    return res


def RFECV_filter(df: DataFrame, y: Series, col_list: List, estimator: Any, keep: float = 0.5, step: int = 1,
                 cv: int = 5) -> List:
    """
    递归特征(交叉验证)消除
    :param df:
    :param y:
    :param col_list:
    :param estimator: 使用的学习器
    :param keep: 保留特征数目或比例
    :param step: 每次递归的步长
    :param cv: 交叉验证折数
    :return:
    """
    if keep >= 1 and isinstance(keep, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(keep, float):
        keep = np.ceil(len(col_list) * keep)

    selector = RFECV(estimator, min_features_to_select=keep, step=step, cv=cv, scoring='roc_auc', n_jobs=-1)
    selector = selector.fit(df[col_list], y)
    mask = selector.get_support()

    res = np.array(col_list)[mask].tolist()

    return res