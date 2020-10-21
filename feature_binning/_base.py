# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/27 11:11
@file: _base.py
@desc: 
"""
from typing import Dict, List

import numpy as np
from pandas import DataFrame, Series

from util import woe_single_all

__SMOOTH__ = 1e-6
__DEFAULT__ = 1e-6

__all__ = ['BinnerMixin', 'encode_woe']


class BinnerMixin:
    def __init__(self, features_info: Dict = None, features_nan_value: Dict = None, max_leaf_nodes: int = 5,
                 min_samples_leaf=0.05):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param features_nan_value: 变量缺失值标识符字典，每个变量可能对应多个缺失值标识符存储为list
        :param max_leaf_nodes: 最大分箱数量
        :param min_samples_leaf: 每个分箱最少样本比例
        """
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.features_info = features_info
        self.features_nan_value = features_nan_value

        self.features_bins = {}  # 每个变量对应分箱结果
        self.features_woes = {}  # 每个变量各个分箱woe值
        self.features_iv = {}  # 每个变量iv值

    def _bin_method(self, x: Series, y: Series, **params):
        """
        获取不同方法的分箱结果
        :param x: 单个变量数据
        :param y: 标签数据
        :param params: 参数
        :return: 分箱区间
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def _bin_threshold(self, x: Series, y: Series, is_num: bool = True, nan_value=None, **params):
        """
        获取单个变量分箱阈值
        :param x: 单个变量数据
        :param y: 标签数据
        :param is_num: 是否为定量变量
        :param nan_value: 缺失值标识符
        :param params: 分箱参数
        :return: 变量分箱区间，缺失值是否单独做为一箱标识
        """
        # 若缺失值标识符为空，定义为-999，因为前面已经用-999填充
        if nan_value is None:
            nan_value = [-999]

        # 判断缺失值数目，如果占比超过min_samples_leaf默认5%, 缺失值单独做为一箱
        flag = 0  # 标识缺失值是否单独做为一箱
        miss_value_num = x.isin(nan_value).sum()
        if miss_value_num > params['min_samples_leaf']:
            params['max_leaf_nodes'] -= 1
            y = y[~x.isin(nan_value)]
            x = x[~x.isin(nan_value)]
            flag = 1

        if is_num:
            bucket = self._bin_method(x, y, **params)
        else:
            bin_map = encode_woe(x, y)
            x = x.map(bin_map)
            bins = self._bin_method(x, y, **params)
            keys = np.array(list(bin_map.keys()))
            values = np.array(list(bin_map.values()))
            bucket = []
            for i in range(len(bins) - 1):
                mask = (values > bins[i]) & (values <= bins[i + 1])
                bucket.append(list(keys[mask]))

        if flag == 1:
            bucket.insert(0, nan_value)

        return bucket, flag

    def _get_woe_iv(self, x: Series, y: Series, col_name=None):
        """
        计算每个分箱指标
        :param x: 单个变量数据
        :param y: 标签数据
        :param col_name: 变量列名
        :return: woe列表，iv值
        """
        is_num = self.features_info[col_name]
        nan_flag = self.features_bins[col_name]['flag']
        bins = self.features_bins[col_name]['bins']
        B = y.sum()
        G = y.size - B
        b_bins = []
        g_bins = []

        if is_num:
            if nan_flag == 1:
                for i in range(len(bins) - 1):
                    if i == 0:
                        mask = x.isin(bins[0])
                    else:
                        mask = (x > bins[i]) & (x <= bins[i + 1]) & (~x.isin(bins[0]))
                    b_bins.append(y[mask].sum())
                    g_bins.append(mask.sum() - y[mask].sum())
            else:
                for i in range(len(bins) - 1):
                    mask = (x > bins[i]) & (x <= bins[i + 1])
                    b_bins.append(y[mask].sum())
                    g_bins.append(mask.sum() - y[mask].sum())
        else:
            for v in bins:
                mask = x.isin(v)
                b_bins.append(y[mask].sum())
                g_bins.append(mask.sum() - y[mask].sum())

        b_bins = np.array(b_bins)
        g_bins = np.array(g_bins)
        woes = woe_single_all(B, G, b_bins, g_bins).tolist()
        temp = (b_bins + __SMOOTH__) / (B + __SMOOTH__) - (g_bins + __SMOOTH__) / (G + __SMOOTH__)
        iv = float(np.around((temp * woes).sum(), 6))

        return woes, iv

    def _get_binning_threshold(self, X: DataFrame, y: Series):
        """
        获取分箱阈值，具体函数参见分箱方法的重写函数
        :param X:
        :param y:
        :return:
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def fit(self, X: DataFrame, y: Series):
        """
        分箱，获取最终结果
        :param X: 所有变量数据
        :param y: 标签数据
        :return:
        """
        # 判断y是否为0,1变量
        assert np.array_equal(y, y.astype(bool)), 'y取值非0,1'
        # 填充空值为-999
        X.fillna(-999, inplace=True)
        # 获取分箱阈值
        self.features_bins = self._get_binning_threshold(X, y)
        # 获取分箱woe值和iv值
        for col in X.columns:
            self.features_woes[col], self.features_iv[col] = self._get_woe_iv(X[col], y, col)


def encode_woe(X: Series, y: Series) -> Dict:
    """
    定性变量woe有序转码，根据woe从小到大排序，替换为0-n数字, 返回转码后对应关系
    :param X: 变量数据
    :param y: y标签数据
    :return: 返回转码后的Dict
    """
    B = y.sum()
    G = y.size - B
    unique_value = X.unique()
    mask = (unique_value.reshape(-1, 1) == X.values)
    mask_bad = mask & (y.values == 1)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    woe_value = woe_single_all(B, G, b, g)
    woe_value_sort = np.argsort(woe_value)
    res = dict(zip(unique_value, woe_value_sort))
    return res


def get_woe_inflexions(woes: List[float]) -> int:
    """
    获取分箱结果拐点数目
    :param woes:
    :return:
    """
    n = len(woes)
    if n <= 2:
        return 0
    return sum(1 if (b - a) * (b - a) > 0 else 0 for a, b, c in zip(woes[:-2], woes[1:-1], woes[2:]))
