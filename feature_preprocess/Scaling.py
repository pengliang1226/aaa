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
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from sklearn.base import TransformerMixin


class FeatureScaling(TransformerMixin):
    def __init__(self):
        """
        特征缩放初始化函数
        """
        self.scaler = None

    def Normalizer_Scaler(self, norm: str = 'l1'):
        """
        正则化缩放，正则化为行操作，它试图“缩放”每个样本，使其具有单位范数。由于正则化在每一行都独立起作用，它会扭曲特征之间的关系，因此较为不常见
        :param norm: 取值 l1, l2, max
        :return:
        """
        self.scaler = Normalizer(norm=norm)

    def MinMax_Scaler(self, feature_range: Any = (0, 1)):
        """
        最大最小缩放, 有离群值时采用robust方法
        :param feature_range: 缩放区间
        :return:
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def MaxAbs_Scaler(self):
        """
        最大缩放, 有离群值时采用robust方法
        :return:
        """
        self.scaler = MaxAbsScaler()

    def Standard_Scaler(self):
        """
        标准缩放（z-score标准化）, 有离群值时采用robust方法
        :return:
        """
        self.scaler = StandardScaler()

    def Robust_Scaler(self, quantile_range: Any = (25.0, 75.0)):
        """
        稳健缩放，可以抗异常值
        :param quantile_range: 分位数
        :return:
        """
        self.scaler = RobustScaler(quantile_range=quantile_range)

    def Power_Transform(self, method: str = 'box-cox'):
        """
        幂次变换，将原始分布转换为正态分布
        :param method: box-cox，变换只适用于正数；yeo-johnson，变换适用于正数和负数
        :return:
        """
        self.scaler = PowerTransformer(method=method, standardize=True)

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