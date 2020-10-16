# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/27 18:51
@file: AdjustBinner.py
@desc: 
"""
from math import inf
from typing import Dict

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

from calculate.feature_binning._base import BinnerMixin


class AdjustBinner(BaseEstimator, BinnerMixin):
    def __init__(self, features_info: Dict = None, is_right: bool = True, bins_threshold: Dict = None):
        """
        初始化函数
        :param features_info: 变量属性类型
        :param is_right: 分箱区间是否右闭
        :param bins_threshold: 变量手动分箱区间
        """
        # basic params
        BinnerMixin.__init__(self, features_info=features_info, is_right=is_right)

        # decision tree params
        self.bins_threshold = bins_threshold

    def _get_binning_threshold(self, X: DataFrame, y: Series) -> Dict:
        """
        获取变量分箱阈值
        :param X:
        :param y:
        :return:
        """
        bins_threshold = {}
        for col, is_num in self.features_info.items():
            bins = str_to_bins(self.bins_threshold[col], is_num=is_num)
            bins_threshold[col] = bins

        return bins_threshold


def str_to_bins(bins: str, is_num: bool = True) -> list:
    """
    将字符串类型的分箱阈值转换为由阈值组成的数组，以便后续分箱
    :param bins: 变量分箱区间，字符串
    :param is_num: 是否为定量变量
    :return:
    """
    bins = bins.replace(" ", "").split(";")
    if is_num:
        bins = [b.replace("(", "").replace(")", "").replace("[", "").replace("]", "") for b in bins]
        res = [-inf]
        for b in bins:
            res.append(float(b.split(",")[1]))
        res[-1] = inf
    else:
        res = [b.replace("[", "").replace("]", "").split(",") for b in bins]
    return res
