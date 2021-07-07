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
import pandas as pd
from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier

from feature_binning._base import BinnerMixin


class DecisionTreeBinner(BinnerMixin):
    def __init__(self, features_info: Dict = None, features_nan_value: Dict = None, max_leaf_nodes=5,
                 min_samples_leaf=0.05, criterion='gini', max_depth=None, min_samples_split=2,
                 random_state=1234, is_psi: int = 0, is_ks: int = 0, is_gini: int = 0):
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
        :param is_psi: 是否是为了计算psi进行的分箱，如果是则不需要进行分箱后的合并操作
        :param is_ks: 是否计算ks
        :param is_gini: 是否计算gini
        """
        # basic params
        BinnerMixin.__init__(self, features_info=features_info, features_nan_value=features_nan_value,
                             max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, is_ks=is_ks,
                             is_gini=is_gini)

        # decision tree params
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.is_psi = is_psi
        self.B_G_rate = None

    def _bin_method(self, X: Series, y: Series, **params) -> list:
        """
        获取决策树分箱结果
        :param X: 单个变量数据
        :param y: 标签数据
        :param params: 决策树参数
        :return: 决策树分箱区间
        """
        # 初步分箱
        tree = DecisionTreeClassifier(**params)
        tree.fit(X.values.reshape(-1, 1), y)
        cutoffs = [-inf]
        cutoffs.extend(np.sort(tree.tree_.threshold[tree.tree_.feature != -2]))
        cutoffs.append(inf)
        cutoffs = np.array(cutoffs)

        if self.is_psi == 0:
            # 分箱中只存在好或坏客户的箱体
            # 首先判断首个区间的方向
            # 如果b/g大于B/G则区间方向定为b/g减小的方向, 如果区间全是好客户则向后合并, 反之向前;
            # 如果b/g大于B/G则区间方向定为b/g增大的方向, 如果区间全是好客户则向前合并, 反之向后;
            x_cut = pd.cut(X, cutoffs, include_lowest=True, labels=False)
            cutoffs = cutoffs[1:]
            freq_tab = pd.crosstab(x_cut, y)
            freq = freq_tab.values
            value_counts = freq[:, 1] / freq[:, 0]
            while np.where(freq == 0)[0].size > 0:
                idx = np.where(freq == 0)[0][0]
                if idx == 0:
                    tmp = freq[idx] + freq[idx + 1]
                    freq[idx + 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx)
                elif idx == len(cutoffs) - 1:
                    tmp = freq[idx - 1] + freq[idx]
                    freq[idx - 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx - 1)
                else:
                    if value_counts[0] >= self.B_G_rate:  # 方向定为b/g减小的方向
                        if np.where(freq[idx] > 0)[0] == 0:  # 区间全为好样本
                            tmp = freq[idx] + freq[idx + 1]
                            freq[idx + 1] = tmp
                            # 删除对应的切分点
                            cutoffs = np.delete(cutoffs, idx)
                        else:
                            tmp = freq[idx - 1] + freq[idx]
                            freq[idx - 1] = tmp
                            # 删除对应的切分点
                            cutoffs = np.delete(cutoffs, idx - 1)
                    else:
                        if np.where(freq[idx] > 0)[0] == 0:  # 区间全为好样本
                            tmp = freq[idx - 1] + freq[idx]
                            freq[idx - 1] = tmp
                            # 删除对应的切分点
                            cutoffs = np.delete(cutoffs, idx - 1)
                        else:
                            tmp = freq[idx] + freq[idx + 1]
                            freq[idx + 1] = tmp
                            # 删除对应的切分点
                            cutoffs = np.delete(cutoffs, idx)

                # 删除idx
                freq = np.delete(freq, idx, 0)
                # 更新value_counts
                value_counts = freq[:, 1] / freq[:, 0]
        else:
            cutoffs = cutoffs[1:]

        threshold = [-inf]
        threshold.extend(cutoffs[:-1].tolist())
        threshold.append(inf)

        return threshold

    def _get_binning_threshold(self, df: DataFrame, y: Series) -> Dict:
        """
        获取分箱阈值
        :param df: 所有变量数据
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
        self.B_G_rate = y.sum() / (y.size - y.sum())
        for col in df.columns:
            feat_type = self.features_info.get(col)
            nan_value = self.features_nan_value.get(col)
            bins, flag = self._bin_threshold(df[col], y, is_num=feat_type, nan_value=nan_value, **params)
            self.features_bins[col] = {'bins': bins, 'flag': flag}
