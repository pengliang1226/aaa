# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/12/3 15:42
@file: Scaling.py
@desc: 
"""
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, MinMaxScaler, StandardScaler, Normalizer
from sklearn.base import TransformerMixin


class FeatureScaling(TransformerMixin):
    def __init__(self):
        """
        特征缩放初始化函数
        """
        self.scaler = None

    def Normalizer_normalization(self, norm: str = 'l1'):
        """
        基础归一化
        :param norm: 取值 l1, l2, max
        :return:
        """
        self.scaler = Normalizer(norm=norm)

    def MinMax_normalization(self, feature_range: Any = (0, 1)):
        """
        归一化, 有离群值时采用robust方法
        :param feature_range: 缩放区间
        :return:
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def MaxAbs_normalization(self):
        """
        归一化, 有离群值时采用robust方法
        :return:
        """
        self.scaler = MaxAbsScaler()

    def Standard_standardization(self):
        """
        z-score标准化, 有离群值时采用robust方法
        :return:
        """
        self.scaler = StandardScaler()

    def Robust_standardization(self, quantile_range: Any = (25.0, 75.0)):
        """
        Robust(鲁棒性)标准化
        :param quantile_range: 分位数
        :return:
        """
        self.scaler = RobustScaler(quantile_range=quantile_range)

    def fit(self, X: DataFrame):
        """
        拟合函数
        :param X:
        :return:
        """
        self.scaler.fit(X)

    def transform(self, X: DataFrame):
        """
        转换函数
        :param X:
        :return:
        """
        res = self.scaler.transform(X)
        res = pd.DataFrame(res, columns=X.columns)

        return res