# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/28 18:27
@file: ScoreStretch.py
@desc: 评分拉伸（预测概率转换为得分）
"""
import numpy as np
from numpy import ndarray
from pandas import DataFrame


class ScoreStretch:
    def __init__(self, theta: float = None, P0: float = None, PDO: float = None, score_max: float = 1000,
                 score_min: float = 300, A: float = None, B: float = None, S: float = None, W: float = None):
        """
        初始化函数
        :param theta: 计算A，B时，需做两个假设，其一odds某个特定违约比率theta下的分数P0; P0 = A - B * log(theta)
        :param P0: 计算A，B时，需做两个假设，其一odds某个特定违约比率theta下的分数P0
        :param PDO: 计算A，B时，需做两个假设，其二该违约比率翻倍时分数的减少值; P0 - PDO = A - B * log(2*theta)
        :param score_max: 分数区间最大值
        :param score_min: 分数区间最小值
        :param A: 分数计算公式的常数；score = A - B * log(odds)
        :param B: 分数计算公式的系数
        :param S: 样本坏客率，可根据样本计算
        :param W: 实际业务数据坏客率，需用户设置
        """
        # default params
        self.theta = theta
        self.P0 = P0
        self.PDO = PDO
        self.score_min = score_min
        self.score_max = score_max
        self.S = S
        assert self.S is not None, '样本坏客率为空'
        self.W = W
        if theta is not None and P0 is not None and PDO is not None:
            self.B = PDO / np.log(2)
            self.A = P0 + self.B * np.log(theta)
        else:
            self.A = A
            self.B = B

    def get_A_B(self, pred: ndarray, extremes: float = 0):
        """
        自适应计算基础分A和翻倍分B
        :param pred
        :param extremes: 极值占比，避免结果受极值影响
        :return:
        """
        odds = np.sort(pred / (1 - pred))
        odds_max = np.percentile(odds, (1 - extremes) * 100)  # 避免极值干扰
        odds_min = np.percentile(odds, extremes * 100)  # 避免极值干扰
        self.B = (self.score_max - self.score_min) / np.log(odds_max / odds_min)
        if self.W is None:
            self.A = self.score_min + self.B * np.log(odds_max / (self.S / (1 - self.S)))
        else:
            self.A = self.score_min + self.B * np.log(odds_max / (self.W / (1 - self.W)))

    def predict(self, X: DataFrame, model=None) -> ndarray:
        """
        预测函数
        :param X: 预测样本
        :param model: 模型
        :return:
        """
        assert model is not None, '模型文件为空'
        model_pred = model.predict_proba(X)[:, 1]

        # 预测概率还权
        if self.W is not None:
            model_pred = 1 / (1 + ((self.S / (1 - self.S)) / (self.W / (1 - self.W))) * (1 / model_pred - 1))

        return model_pred

    def transform(self, pred: ndarray) -> ndarray:
        """
        预测概率根据实际业务进行还权，并转换为评分
        :return:
        """
        pred[pred == 1] = 0.9999
        pred[pred == 0] = 0.0001

        # 计算基础分翻倍分
        if self.A is None and self.B is None:
            self.get_A_B(pred)

        if self.A is None or self.B is None:
            raise Exception('基础分, 翻倍分存在空值')
        # 概率转换为得分
        if self.W is None:
            score = self.A - self.B * np.log((pred / (1 - pred)) / (self.S / (1 - self.S)))
        else:
            score = self.A - self.B * np.log((pred / (1 - pred)) / (self.W / (1 - self.W)))
        # 分数限制范围
        if self.score_max is not None:
            score[score > self.score_max] = self.score_max
        if self.score_min is not None:
            score[score < self.score_min] = self.score_min

        score = np.around(score, 2)

        return score

    def predict_transform(self, X: DataFrame, model=None) -> ndarray:
        pred = self.predict(X, model)
        score = self.transform(pred)
        return score
