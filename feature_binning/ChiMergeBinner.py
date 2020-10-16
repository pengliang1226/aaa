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

from calculate.feature_binning._base import BinnerMixin, encode_woe


class ChiMergeBinner(BinnerMixin):
    def __init__(self, features_info: Dict = None, is_right: bool = True, max_leaf_nodes=5,
                 min_samples_leaf=0.05, confidence_level=0.95):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param is_right: 分箱区间是否右闭
        :param max_leaf_nodes: 最大分箱数量
        :param min_samples_leaf: 每个分箱最少样本量
        :param confidence_level: 置信度
        """
        # basic params
        BinnerMixin.__init__(self, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                             features_info=features_info, is_right=is_right)

        # chi params
        self.confidence_level = confidence_level

    def _bin_chi2(self, x: Series, y: Series, is_num: bool = True):
        """
        计算单个变量分箱阈值
        :param x: 单个变量数据
        :param y: 标签数据
        :param is_num: 是否为定量变量
        :return:
        """
        if is_num:
            return chiMerge(x, y, max_interval=self.max_leaf_nodes, confidence_level=self.confidence_level)
        else:
            bin_map = encode_woe(x, y)
            temp_x = x.map(bin_map)
            bins = chiMerge(temp_x, y, max_interval=self.max_leaf_nodes, confidence_level=self.confidence_level)
            keys = np.array(list(bin_map.keys()))
            values = np.array(list(bin_map.values()))
            bucket = []
            for i in range(len(bins) - 1):
                mask = (values > bins[i]) & (values <= bins[i + 1])
                bucket.append(list(keys[mask]))
            return bucket

    def _get_binning_threshold(self, X: DataFrame, y: Series):
        """
        获取变量分箱阈值
        :param X: 所有变量数据
        :param y: 标签数据
        :return:
        """
        bins_threshold = {}
        for col, is_num in self.features_info.items():
            if X[col].unique().size > 1:
                bins = self._bin_chi2(X[col], y, is_num=is_num)
            else:
                bins = [-inf, inf] if is_num else [list(X[col].unique())]
            bins_threshold[col] = bins

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


def chiMerge(x: Series, y: Series, max_interval: int = 10, confidence_level: float = 0.95) -> List:
    """
    卡方分箱--卡方阈值，最大分箱数同时限制
    :param x: 单个变量数据
    :param y: 标签数据
    :param max_interval: 最大分箱数量
    :param confidence_level: 置信度
    :return:
    """
    freq_tab = pd.crosstab(x, y)
    freq = freq_tab.values
    cutoffs = freq_tab.index.values
    # 卡方阈值
    threshold = chi2.isf(1 - confidence_level, y.unique().size - 1)

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
        if minvalue < threshold or len(freq) > max_interval:
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

    res = [-inf]
    res.extend(cutoffs[:-1])
    res.append(inf)
    return res
