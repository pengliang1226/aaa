# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:46
@file: FilterMethod.py
@desc: 过滤法，先对数据集进行特征选择，其过程与后续学习器无关，即设计一些统计量来过滤特征，并不考虑后续学习器问题。
"""
from typing import List

import numpy as np
from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold


def variance_filter(df: DataFrame, col_list: List, threshold: float = 0.65) -> List:
    """
    方差筛选
    :param df:
    :param col_list:
    :param threshold:
    :return:
    """


def correlation_filter(df: DataFrame, col_list: List, threshold: float = 0.65) -> List:
    """
    相关性变量筛选, 默认选用Pearson, 字符型特征在使用该函数前先进行转码
    :param df: 数据
    :param col_list: 变量列表，按iv值从大到小排序
    :param threshold: 相关系数阈值
    :return:
    """
    data_array = np.array(df[col_list])
    corr_result = np.fabs(np.corrcoef(data_array.T))

    idx = []
    res = []
    for i, col in enumerate(col_list):
        if i == 0:
            idx.append(i)
            res.append(col)
        else:
            corr = corr_result[i, idx]
            if (corr < threshold).all():
                idx.append(i)
                res.append(col)

    return res


def vif_filter(df: DataFrame, col_list: List, threshold=10) -> List:
    """
    多重共线性筛选，逐步剔除vif大于阈值的变量，直至所有变量的vif值小于阈值
    :param df: 数据
    :param col_list: 变量列表，按iv值从大到小排序
    :param threshold: vif阈值，大于该值表示存在多重共线性
    :return:
    """
    vif_array = np.array(df[col_list])
    vifs_list = [variance_inflation_factor(vif_array, i) for i in range(vif_array.shape[1])]
    vif_high = [k for k, v in zip(col_list, vifs_list) if v >= threshold]
    if len(vif_high) > 0:
        for col in reversed(vif_high):
            col_list.remove(col)
            vif_array = np.array(df[col_list])
            vifs_list = [variance_inflation_factor(vif_array, i) for i in range(vif_array.shape[1])]
            if (np.array(vifs_list) < threshold).all():
                break

    return col_list
