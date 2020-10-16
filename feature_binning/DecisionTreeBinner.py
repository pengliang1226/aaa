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
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from calculate.feature_binning._base import BinnerMixin, encode_woe


class DecisionTreeBinner(BaseEstimator, BinnerMixin):
    def __init__(self, features_info: Dict = None, is_right: bool = True,
                 max_leaf_nodes=5, min_samples_leaf=0.05, criterion='gini',
                 max_depth=None, min_samples_split=2, random_state=1234):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param is_right: 分箱区间是否右闭
        :param max_leaf_nodes: 最大分箱数量
        :param min_samples_leaf: 每个分箱最少样本量
        :param criterion: 决策树分裂指标
        :param max_depth: 最大深度
        :param min_samples_split: 决策树节点分裂最少样本量
        :param random_state: 随机种子
        """
        # basic params
        BinnerMixin.__init__(self, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                             features_info=features_info, is_right=is_right)
        # decision tree params
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def _get_binning_threshold(self, X: DataFrame, y: Series) -> Dict:
        """
        获取分箱阈值
        :param X: 所有变量数据
        :param y: 标签数据
        :return:
        """
        params = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": max(self.min_samples_, 50),
            "max_leaf_nodes": self.max_leaf_nodes,
            "random_state": self.random_state
        }
        bins_threshold = {}
        for col, is_num in self.features_info.items():
            bins = dt_binning(X[col], y, is_num=is_num, **params)
            bins_threshold[col] = bins

        return bins_threshold


def dt_threshold(x: Series, y: Series, **params) -> list:
    """
    获取决策树阈值
    :param x: 单个变量数据
    :param y: 标签数据
    :param params: 决策树参数
    :return:
    """
    tree = DecisionTreeClassifier(**params)
    tree.fit(x.values.reshape(-1, 1), y)
    threshold = [-inf]
    threshold.extend(np.sort(tree.tree_.threshold[tree.tree_.feature != -2]).tolist())
    threshold.append(inf)
    return threshold


def dt_binning(x: Series, y: Series, is_num: bool = True, **params) -> list:
    """
    使用决策树进行最优分箱
    :param x: 单个变量数据
    :param y: 标签数据
    :param is_num: 是否为定量变量
    :param params: 决策树参数
    :return:
    """
    if is_num:
        return dt_threshold(x, y, **params)
    else:
        bin_map = encode_woe(x, y)
        temp_x = x.map(bin_map)
        bins = dt_threshold(temp_x, y, **params)
        keys = np.array(list(bin_map.keys()))
        values = np.array(list(bin_map.values()))
        bucket = []
        for i in range(len(bins) - 1):
            mask = (values > bins[i]) & (values <= bins[i + 1])
            bucket.append(list(keys[mask]))

        return bucket
