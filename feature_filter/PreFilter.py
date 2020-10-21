# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:46
@file: PreFilter.py
@desc: 变量预筛选
"""
from typing import Dict, List

import numpy as np
from pandas import DataFrame


def threshold_filter(X: DataFrame, null_threshold: float = 0.8, mode_threshold: float = 0.9,
                     unique_threshold: float = 0.8, null_flag: Dict = None) -> List:
    """
    根据缺失值比例、同值占比、唯一值占比筛选
    :param X: 数据
    :param null_threshold: 缺失值占比阈值
    :param mode_threshold: 同值占比阈值
    :param unique_threshold: 唯一值占比阈值
    :param null_flag: 缺失值标识符, list可能存在多个缺失值
    :return: 返回需要剔除的变量
    """
    res = []
    for col in X.columns:
        col_data = X[col]
        if null_flag is not None:
            null_value = null_flag.get(col)
            col_data.replace(null_value, [np.nan] * len(null_value), inplace=True)

        # 缺失值判断
        null_rate = col_data.isna().sum() / col_data.shape[0]
        if null_rate > null_threshold:
            res.append(col)
            continue

        # 同值占比判断
        mode = col_data.mode()[0]
        mode_rate = (col_data == mode).sum() / col_data.dropna().shape[0]
        if mode_rate > mode_threshold:
            res.append(col)
            continue

        # 唯一值占比判断
        unique_rate = col_data.dropna().unique().size / col_data.dropna().shape[0]
        if unique_rate > unique_threshold:
            res.append(col)
            continue

    return res
