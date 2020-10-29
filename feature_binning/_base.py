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

from util import woe_single_all, woe_single

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
        assert features_nan_value is not None, '变量缺失值标识符字典为空'
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
            bucket = [[bucket[i], bucket[i + 1]] for i in range(len(bucket) - 1)]
        else:
            bin_map = encode_woe(x, y)
            x = x.map(bin_map)
            bins = self._bin_method(x, y, **params)
            keys = np.array(list(bin_map.keys()))
            values = np.array(list(bin_map.values()))
            bucket = []
            for i in range(len(bins) - 1):
                mask = (values > bins[i]) & (values <= bins[i + 1])
                bucket.append(keys[mask].tolist())

        if flag == 1:
            bucket.insert(0, nan_value)

        return bucket, flag

    def _get_woe_iv(self, x: Series, y: Series, col_name):
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

        if nan_flag == 1:
            mask = x.isin(bins[0])
            b_bins.append(y[mask].sum())
            g_bins.append(mask.sum() - y[mask].sum())
            bins = bins[1:]
            x = x[~mask]
            y = y[~mask]

        if is_num:
            for left, right in bins:
                mask = (x > left) & (x <= right)
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
        # 判断数据是否存在缺失
        assert any(X.isna()), '数据存在空值'
        # 获取分箱阈值
        self._get_binning_threshold(X.copy(), y.copy())
        # 获取分箱woe值和iv值
        for col in X.columns:
            self.features_woes[col], self.features_iv[col] = self._get_woe_iv(X[col].copy(), y.copy(), col)

    def binning_trim(self, X: DataFrame, y: Series, col_list: List):
        """
        分箱单调性调整，并重新计算相关指标
        :param X: 数据
        :param y: y标签数据
        :param col_list: 需要调整分箱的列
        :return:
        """
        B = y.sum()
        G = y.size - B
        for col in col_list:
            col_data = X[col].copy()
            y_data = y.copy()
            feat_type = self.features_info[col]
            bins = self.features_bins[col]['bins']
            flag = self.features_bins[col]['flag']
            woes = np.array(self.features_woes[col])

            # 剔除缺失值相关信息
            if flag == 1:
                mask = col_data.isin(bins[0])
                col_data = col_data[~mask]
                y_data = y_data[~mask]
                woes = woes[1:]
                bins = bins[1:]

            # 判断是否需要调整woe单调性
            if get_woe_inflexions(woes) == 0:
                continue

            # 开始调整，分单调递减和单调递增, 合并区间时向前合并
            while get_woe_inflexions(woes) > 0:
                # 判断woes开始位置值正负来确定不符合单调性的位置
                if woes[0] > 0:
                    idx = np.where(woes[:-1] < woes[1:])[0][0] + 1
                elif woes[0] < 0:
                    idx = np.where(woes[:-1] > woes[1:])[0][0] + 1
                else:
                    break
                # 重新计算合并后的woe值
                if feat_type == 1:
                    mask = (col_data > bins[idx - 1][0]) & (col_data <= bins[idx][1])
                    bins[idx - 1][1] = bins[idx][1]
                else:
                    mask = (col_data.isin(bins[idx - 1])) | (col_data.isin(bins[idx]))
                    bins[idx - 1] += bins[idx]
                b = y_data[mask].sum()
                g = mask.sum() - b
                woes[idx - 1] = woe_single(B, G, b, g)
                woes = np.delete(woes, idx)
                del bins[idx]

                # 如果区间个数小于等于2，退出
                if len(woes) <= 2:
                    break

            # 修改原始分箱信息
            if flag == 1:
                self.features_bins[col]['bins'][1:] = bins
            else:
                self.features_bins[col]['bins'] = bins

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
    return sum(1 if (b - a) * (b - c) > 0 else 0 for a, b, c in zip(woes[:-2], woes[1:-1], woes[2:]))
