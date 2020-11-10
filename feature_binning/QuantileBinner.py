# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/20 17:30
@file: QuantileBinner.py
@desc:
"""
from typing import Dict
from math import inf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from feature_binning._base import BinnerMixin


class QuantileBinner(BinnerMixin):
    def __init__(self, features_info: Dict = None, features_nan_value: Dict = None, max_leaf_nodes=5,
                 min_samples_leaf=0.05, is_psi: int = 0):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param features_nan_value: 变量缺失值标识符字典，每个变量可能对应多个缺失值标识符存储为list
        :param max_leaf_nodes: 分箱数量
        :param min_samples_leaf: 每个分箱最少样本比例
        :param is_psi: 是否是为了计算psi进行的分箱，如果是则不需要进行分箱后的合并操作
        """
        # basic params
        BinnerMixin.__init__(self, features_info=features_info, features_nan_value=features_nan_value,
                             max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
        self.is_psi = is_psi

    def _bin_method(self, X: Series, y: Series, **params) -> list:
        """
        获取等频分箱结果
        :param X: 单个变量数据
        :param y: 标签数据
        :param params: 决策树参数
        :return: 等频分箱区间
        """
        # 初步分箱
        if X.unique().size <= params['max_leaf_nodes']:  # 如果变量唯一值个数小于分箱数, 则直接按唯一值作为阈值
            cutoffs = [-inf]
            cutoffs.extend(np.sort(X.unique()))
            cutoffs = np.array(cutoffs)
        else:
            _, cutoffs = pd.qcut(X, params['max_leaf_nodes'], duplicates='drop', retbins=True)

        if self.is_psi == 0:
            # 分箱中只存在好或坏客户的箱体, 与相邻区间样本数目少的合并
            x_cut = pd.cut(X, cutoffs, include_lowest=True, labels=False)
            cutoffs = cutoffs[1:]
            freq_tab = pd.crosstab(x_cut, y)
            freq = freq_tab.values
            value_counts = np.sum(freq, axis=1)
            while np.where(freq == 0)[0].size > 0:
                idx = np.where(freq == 0)[0][0]
                left = value_counts[idx - 1] if idx > 0 else inf
                right = value_counts[idx + 1] if idx < len(cutoffs) - 1 else inf
                if left > right:
                    tmp = freq[idx] + freq[idx + 1]
                    freq[idx + 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx)
                else:
                    tmp = freq[idx - 1] + freq[idx]
                    freq[idx - 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx - 1)
                # 删除idx
                freq = np.delete(freq, idx, 0)
                # 更新count值
                value_counts = np.sum(freq, axis=1)

            # # 分箱中客户数量小于阈值的箱体, 向前合并
            # x_cut = pd.cut(x, threshold, include_lowest=True, labels=False)
            # print(x_cut.value_counts(sort=False))
            # index = np.where(x_cut.value_counts(sort=False) < params['min_samples_leaf'])[0]
            # if index.size > 0:
            #     threshold = [threshold[i] for i in range(len(threshold)) if i not in index]

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
            "min_samples_leaf": max(int(np.ceil(y.size * self.min_samples_leaf)), 50),
            "max_leaf_nodes": self.max_leaf_nodes
        }

        for col in df.columns:
            feat_type = self.features_info.get(col)
            nan_value = self.features_nan_value.get(col)
            bins, flag = self._bin_threshold(df[col], y, is_num=feat_type, nan_value=nan_value, **params)
            self.features_bins[col] = {'bins': bins, 'flag': flag}
