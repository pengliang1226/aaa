# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/20 17:30
@file: QuantileBinner.py
@desc:
"""
from math import inf
from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from feature_binning._base import BinnerMixin


class QuantileBinner(BinnerMixin):
    def __init__(self, features_info: Dict = None, features_nan_value: Dict = None, max_leaf_nodes=5,
                 min_samples_leaf=0.05):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param features_nan_value: 变量缺失值标识符字典，每个变量可能对应多个缺失值标识符存储为list
        :param max_leaf_nodes: 分箱数量
        :param min_samples_leaf: 每个分箱最少样本比例
        """
        # basic params
        BinnerMixin.__init__(self, features_info=features_info, features_nan_value=features_nan_value,
                             max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)

    def _bin_method(self, x: Series, y: Series, **params) -> list:
        """
        获取等频分箱结果
        :param x: 单个变量数据
        :param y: 标签数据
        :param params: 决策树参数
        :return: 等频分箱区间
        """
        _, bins = pd.qcut(x, params['max_leaf_nodes'], duplicates='drop', retbins=True)
        threshold = [-inf]
        threshold.extend(bins[1:-1].tolist())
        threshold.append(inf)
        return threshold

    def _get_binning_threshold(self, X: DataFrame, y: Series) -> Dict:
        """
        获取分箱阈值
        :param X: 所有变量数据
        :param y: 标签数据
        :return: 变量分箱区间字典
        """
        params = {
            "min_samples_leaf": max(int(np.ceil(y.size * self.min_samples_leaf)), 50),
            "max_leaf_nodes": self.max_leaf_nodes
        }
        bins_threshold = {}
        for col in X.columns:
            feat_type = self.features_info.get(col)
            nan_value = self.features_nan_value.get(col) if self.features_nan_value is not None else None
            bins, flag = self._bin_threshold(X[col], y, is_num=feat_type, nan_value=nan_value, **params)
            bins_threshold[col] = {'bins': bins, 'flag': flag}

        return bins_threshold
