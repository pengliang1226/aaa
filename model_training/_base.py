# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/25 15:35
@file: _base.py
@desc: 模型训练基类
"""
from pandas import DataFrame, Series


class TrainerMixin:
    def __init__(self):
        """
        初始化函数
        """
        self.estimator = None

    def create_estimator(self):
        """
        创建学习器
        :return:
        """
        pass

    def fit(self, X: DataFrame, y: Series):
        """
        拟合学习器
        :param X:
        :param y:
        :return:
        """
        self.estimator.fit(X, y)

    def predict(self, X: DataFrame):
        """
        预测类别
        :param X:
        :return:
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: DataFrame):
        """
        预测概率
        :param X:
        :return:
        """
        return self.estimator.predict_proba(X)[:, 1]
