# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/28 9:50
@file: ChiMergeBinner.py
@desc: 
"""
from math import inf
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import chi2

from feature_binning._base import BinnerMixin, encode_woe


class ChiMergeBinner(BinnerMixin):
    def __init__(self, features_info: Dict = None, features_nan_value: Dict = None, max_leaf_nodes=5,
                 min_samples_leaf=0.05, confidence_level=0.95):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param features_nan_value: 变量缺失值标识符字典，每个变量可能对应多个缺失值标识符存储为list
        :param max_leaf_nodes: 最大分箱数量
        :param min_samples_leaf: 每个分箱最少样本比例
        :param confidence_level: 置信度
        """
        # basic params
        BinnerMixin.__init__(self, features_info=features_info, features_nan_value=features_nan_value,
                             max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)

        # chi params
        self.confidence_level = confidence_level

    def _bin_method(self, x: Series, y: Series, **params) -> List:
        """
        获取卡方分箱结果
        :param x: 单个变量数据
        :param y: 标签数据
        :param params: 参数
        :return: 卡方分箱区间
        """
        freq_tab = pd.crosstab(x, y)
        freq = freq_tab.values
        cutoffs = freq_tab.index.values

        chi_results = []
        for i in range(len(freq) - 1):
            v = cal_chi2(freq[i:i + 2])
            chi_results.append(v)
        chi_results = np.array(chi_results)

        while True:
            minidx = np.argmin(chi_results)
            minvalue = np.min(chi_results)
            chi_results = np.delete(chi_results, minidx)
            # 如果最小卡方值小于阈值或箱体数大于最大箱体数，则合并最小卡方值的相邻两组，并继续循环
            if (minvalue < params['threshold'] or len(freq) > params['max_leaf_nodes']) and cutoffs.size > 2:
                # minidx合并到minidx+1
                tmp = freq[minidx] + freq[minidx + 1]
                freq[minidx + 1] = tmp
                # 删除minidx
                freq = np.delete(freq, minidx, 0)
                # 删除对应的切分点
                cutoffs = np.delete(cutoffs, minidx)
                if minidx == 0:
                    chi_results[minidx] = cal_chi2(freq[minidx:minidx + 2])
                elif minidx == len(chi_results):
                    chi_results[minidx - 1] = cal_chi2(freq[minidx - 1:])
                else:
                    chi_results[minidx] = cal_chi2(freq[minidx:minidx + 2])
                    chi_results[minidx - 1] = cal_chi2(freq[minidx - 1:minidx + 1])
            else:
                break

        threshold = [-inf]
        threshold.extend(cutoffs[:-1].tolist())
        threshold.append(inf)
        return threshold

    def _get_binning_threshold(self, X: DataFrame, y: Series) -> Dict:
        """
        获取变量分箱阈值
        :param X: 所有变量数据
        :param y: 标签数据
        :return: 变量分箱区间字典
        """
        params = {
            # 卡方阈值
            "threshold": chi2.isf(1 - self.confidence_level, y.unique().size - 1),
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": max(int(np.ceil(y.size * self.min_samples_leaf)), 50)
        }
        bins_threshold = {}
        for col in X.columns:
            feat_type = self.features_info.get(col)
            nan_value = self.features_nan_value.get(col) if self.features_nan_value is not None else None
            bins, flag = self._bin_threshold(X[col], y, is_num=feat_type, nan_value=nan_value, **params)
            bins_threshold[col] = {'bins': bins, 'flag': flag}

        return bins_threshold


def cal_chi2(arr: ndarray) -> float:
    """
    计算卡方值
    :param arr:
    :return:
    """
    # 计算每行总频数
    R_N = arr.sum(axis=1)
    # 每列总频数
    C_N = arr.sum(axis=0)
    # 总频数
    N = arr.sum()
    # 计算期望频数 C_i * R_j / N。
    E = np.ones(arr.shape) * C_N / N
    E = (E.T * R_N).T
    square = (arr - E) ** 2 / E
    # 期望频数为0时，做除数没有意义，不计入卡方值
    square[E == 0] = 0
    # 卡方值
    v = square.sum()
    return v
