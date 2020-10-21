# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/27 15:31
@file: DecisionTreeBinner.py
@desc: 
"""
from math import inf
from typing import Dict

import numpy as np
from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier

from feature_binning._base import BinnerMixin, encode_woe


class DecisionTreeBinner(BinnerMixin):
    def __init__(self, features_info: Dict = None, features_nan_value: Dict = None, max_leaf_nodes=5,
                 min_samples_leaf=0.05, criterion='gini', max_depth=None, min_samples_split=2,
                 random_state=1234):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param features_nan_value: 变量缺失值标识符字典，每个变量可能对应多个缺失值标识符存储为list
        :param max_leaf_nodes: 最大分箱数量
        :param min_samples_leaf: 每个分箱最少样本比例
        :param criterion: 决策树分裂指标
        :param max_depth: 最大深度
        :param min_samples_split: 决策树节点分裂最少样本量
        :param random_state: 随机种子
        """
        # basic params
        BinnerMixin.__init__(self, features_info=features_info, features_nan_value=features_nan_value,
                             max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)

        # decision tree params
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def _bin_method(self, x: Series, y: Series, **params) -> list:
        """
        获取决策树分箱结果
        :param x: 单个变量数据
        :param y: 标签数据
        :param params: 决策树参数
        :return: 决策树分箱区间
        """
        tree = DecisionTreeClassifier(**params)
        tree.fit(x.values.reshape(-1, 1), y)
        threshold = [-inf]
        threshold.extend(np.sort(tree.tree_.threshold[tree.tree_.feature != -2]).tolist())
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
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": max(int(np.ceil(y.size * self.min_samples_leaf)), 50),
            "max_leaf_nodes": self.max_leaf_nodes,
            "random_state": self.random_state
        }
        bins_threshold = {}
        for col in X.columns:
            feat_type = self.features_info.get(col)
            nan_value = self.features_nan_value.get(col) if self.features_nan_value is not None else None
            bins, flag = self._bin_threshold(X[col], y, is_num=feat_type, nan_value=nan_value, **params)
            bins_threshold[col] = {'bins': bins, 'flag': flag}

        return bins_threshold



