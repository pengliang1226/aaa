# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/27 11:11
@file: _base.py
@desc: 
"""
from typing import Dict, Union, List, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_numeric_dtype

from calculate.util.calculate_util import woe_single_all

__SMOOTH__ = 1e-6
__DEFAULT__ = 1e-6

__all__ = ['BinnerMixin', 'encode_woe']


# TODO: 只能对数值型变量分箱
class BinnerMixin:
    def __init__(self, max_leaf_nodes: int = 5, min_samples_leaf: float = 0.05,
                 features_info: Dict = None, is_right: bool = True, bad_y: Any = 1):
        """
        初始化函数
        :param max_leaf_nodes: 最大分箱数量
        :param min_samples_leaf: 每个分箱最少样本量
        :param features_info: 变量属性类型；定性、定量
        :param is_right: 区间是否右闭
        :param bad_y: 坏样本值
        """
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.features_info = {k: True if v == '1' else 0 for k, v in features_info.items()}
        self.is_right = is_right
        self.bad_y = bad_y

        self.features_ = list(features_info.keys())
        self.min_samples_ = None
        self.results_ = {}

    def _get_binning_threshold(self, X: DataFrame, y: Series):
        """
        获取分箱阈值，具体函数参见分箱方法的重写函数
        :param X:
        :param y:
        :return:
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def _calc_res(self, x: Series, y: Series, bins: List, is_num: bool = True) -> Dict:
        """
        计算每个分箱指标
        :param x: 单个变量数据
        :param y: 标签数据
        :param bins: 分箱阈值
        :param is_num: 是否为定量变量
        :return:
        """
        count_bins = {}
        bad_bins = {}
        if is_num:
            x = pd.cut(x, bins=bins, right=self.is_right, labels=False)
            uni_v = [i for i in range(len(bins) - 1)]
            for idx, v in enumerate(uni_v):
                mask = (x == v)
                count_bins[v] = mask.sum()
                bad_bins[v] = y[mask].sum()
            temp_x = x
        else:
            uni_v = bins
            temp_x = pd.Series(np.zeros_like(x).astype(float))
            for idx, v in enumerate(uni_v):
                mask = x.isin(v)
                k = "[" + ",".join([str(i) for i in v]) + "]"
                count_bins[k] = mask.sum()
                bad_bins[k] = y[mask].sum()
                temp_x[mask] = idx

        woe_bins = woe_part(temp_x, y)
        iv_bins = iv_part(temp_x, y)
        res = {"bins": bins_to_str(bins, right=self.is_right, is_num=is_num),
               "count_bins": [str(i) for i in count_bins.values()],
               "bad_bins": [str(i) for i in bad_bins.values()],
               "woe_bins": [format(float(woe_bins[i]), '.6f') if i in woe_bins else str(__DEFAULT__) for i in
                            range(len(count_bins))],
               "iv_bins": [format(float(iv_bins[i]), '.6f') if i in iv_bins else str(0) for i in
                           range(len(count_bins))],
               "group": [str(i) for i in range(len(count_bins))],
               "feat_type": '1' if is_num else 0
               }
        return res

    def fit(self, X: DataFrame, y: Series):
        """
        分箱，获取最终结果
        :param X: 所有变量数据
        :param y: 标签数据
        :return:
        """
        # 判断数据是否为数值型
        assert all(is_numeric_dtype(X[f]) for f in self.features_), 'X should be all of numeric dtypes!'
        # 将y标签列转换为0,1
        y = y.apply(lambda x: 1 if x == self.bad_y else 0)
        # 计算每个分箱最小样本量
        self.min_samples_ = int(np.ceil(y.size * self.min_samples_leaf))
        # 获取分箱阈值
        bins_threshold = self._get_binning_threshold(X, y)
        # 计算最终结果
        for col, bins in bins_threshold.items():
            self.results_[col] = self._calc_res(X[col], y, bins, self.features_info[col])


def encode_woe(x: Series, y: Series) -> Dict:
    """
    对类别变量进行排序，以便分箱，返回的是变量取值和woe的映射关系
    :param x: 单个变量数据
    :param y: 标签数据
    :return:
    """
    B = y.sum()
    G = y.size - B
    unique_value = x.unique()
    mask = (unique_value.reshape(-1, 1) == x.values)
    mask_bad = mask & (y.values == 1)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    woe_value = woe_single_all(B, G, b, g)
    values_sort = np.argsort(woe_value)
    bin_map = dict(zip(np.unique(x), values_sort))
    return bin_map


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


def bins_to_str(bins: List, right: bool = True, is_num: bool = True) -> List:
    """
    将分箱阈值转换为字符串，以便输出
    :param bins: 分箱阈值
    :param right:
    :param is_num:
    :return:
    """
    res = []
    if is_num:
        for i in range(1, len(bins)):
            if right:
                temp = "({}, {}]".format(bins[i - 1], bins[i])
            else:
                temp = "[{}, {})".format(bins[i - 1], bins[i])
            res.append(temp)
    else:
        for v in bins:
            temp = "[{}]".format(",".join([str(i) for i in v]))
            res.append(temp)
    return res


def iv_part(x: Union[Series, ndarray], y: Union[Series, ndarray]) -> Dict:
    """
    计算单个变量中每个分箱的iv值
    :param x: 单个变量数据
    :param y: 标签数据
    :return:
    """
    if isinstance(x, Series):
        x = x.values
    if isinstance(y, Series):
        y = y.values
    B = y.sum()
    G = y.size - B
    unique_value = np.unique(x)
    mask = (unique_value.reshape(-1, 1) == x)
    mask_bad = mask & (y == 1)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    temp = (b + __SMOOTH__) / (B + __SMOOTH__) - (g + __SMOOTH__) / (G + __SMOOTH__)
    iv_value = np.around(temp * woe_single_all(B, G, b, g), 6)
    iv_value = iv_value.tolist()
    res = dict(zip(unique_value, iv_value))
    return res


def woe_part(x: Union[Series, ndarray], y: Union[Series, ndarray]) -> Dict:
    """
    计算单个变量每个分箱的woe
    :param x: 单个变量数据
    :param y: 标签数据
    :return:
    """
    if isinstance(x, Series):
        x = x.values
    if isinstance(y, Series):
        y = y.values
    B = y.sum()
    G = y.size - B
    unique_value = np.unique(x)
    mask = (unique_value.reshape(-1, 1) == x)
    mask_bad = mask & (y == 1)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    woe_value = np.around(woe_single_all(B, G, b, g), 6)
    woe_value = woe_value.tolist()
    res = dict(zip(unique_value, woe_value))
    return res
