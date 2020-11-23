# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/28 18:27
@file: ScoreStretch.py
@desc: 评分拉伸（预测概率转换为得分）
"""
from typing import Dict, Union, List, Set

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame


class ScoreStretch:
    def __init__(self, S: float = None, W: float = None, theta: float = None, P0: float = None, PDO: float = None,
                 score_max: float = 1000, score_min: float = 300, pred: ndarray = None, extremes: float = 0,
                 A: float = None, B: float = None):
        """
        初始化函数
        [注]：计算方式
        1、通过theta，P0，PDO计算，公式为
            P0 = A - B * log(theta)
            P0 - PDO = A - B * log(2*theta)
        2、通过score_max，score_min，pred，extremes计算，公式为
            odds = pred / (1 - pred)
            score_max = A - B * log(odds_min)
            score_min = A - B * log(odds_max)
        :param S: 样本坏客率，可根据样本计算
        :param W: 实际业务数据坏客率，需用户设置
        :param theta: 计算A，B时，需做两个假设，其一odds某个特定违约比率theta下的分数P0; P0 = A - B * log(theta)
        :param P0: 计算A，B时，需做两个假设，其一odds某个特定违约比率theta下的分数P0
        :param PDO: 计算A，B时，需做两个假设，其二该违约比率翻倍时分数的减少值; P0 - PDO = A - B * log(2*theta)
        :param score_max: 分数区间最大值
        :param score_min: 分数区间最小值
        :param pred: 用于计算A，B训练集预测概率
        :param extremes: 避免极端值对计算odds_max和odds_min的影响
        :param A: 分数计算公式的常数；score = A - B * log(odds)
        :param B: 分数计算公式的系数
        """
        # default params
        self.theta = theta
        self.P0 = P0
        self.PDO = PDO
        self.score_min = score_min
        self.score_max = score_max
        self.pred = pred
        self.extremes = extremes
        self.S = S
        assert self.S is not None, '样本坏客率为空'
        self.W = W
        self.A = A
        self.B = B
        self.get_A_B()

    def get_A_B(self):
        """
        自适应计算基础分A和翻倍分B
        :return:
        """
        if self.theta is not None and self.P0 is not None and self.PDO is not None:
            self.B = self.PDO / np.log(2)
            self.A = self.P0 + self.B * np.log(self.theta)
        elif self.score_max is not None and self.score_max is not None and self.pred is not None:
            if self.W is not None:
                self.pred = 1 / (1 + ((self.S / (1 - self.S)) / (self.W / (1 - self.W))) * (1 / self.pred - 1))
            odds = np.sort(self.pred / (1 - self.pred))
            odds_max = np.percentile(odds, (1 - self.extremes) * 100)  # 避免极值干扰
            odds_min = np.percentile(odds, self.extremes * 100)  # 避免极值干扰
            self.B = (self.score_max - self.score_min) / np.log(odds_max / odds_min)
            if self.W is None:
                self.A = self.score_min + self.B * np.log(odds_max / (self.S / (1 - self.S)))
            else:
                self.A = self.score_min + self.B * np.log(odds_max / (self.W / (1 - self.W)))
        elif self.A is None or self.B is None:
            raise Exception('A,B存在缺失值，且无法通过计算得到')
        else:
            pass

    def transform_pred_to_score(self, pred: ndarray) -> ndarray:
        """
        预测概率根据实际业务进行还权，并转换为评分
        :param pred: 预测概率
        :return:
        """
        # 预测概率还权
        if self.W is not None:
            pred = 1 / (1 + ((self.S / (1 - self.S)) / (self.W / (1 - self.W))) * (1 / pred - 1))

        pred[pred == 1] = 0.9999
        pred[pred == 0] = 0.0001

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

    def transform_data_to_score(self, df: DataFrame, estimator) -> ndarray:
        """
        预测概率根据实际业务进行还权，并转换为评分
        :param df: 数据, 变量顺序需与入模顺序一致, 已进行woe转换
        :param estimator: 模型
        :return:
        """
        coef = estimator.coef_[0]
        intercept = estimator.intercept_[0]
        base_score = self.A - self.B * intercept

        for i, col in enumerate(df.columns):
            df[col] = df[col].apply(lambda x: -self.B * coef[i] * x)

        score = np.sum(df.values, axis=1)
        score = np.around(score + base_score, 2)

        return score

    def get_scorecard(self, model_feats: Union[List, Set], estimator, bins_info: Dict) -> DataFrame:
        """
        获取评分卡表格
        :param model_feats: 入模变量
        :param estimator: 模型
        :param bins_info: 包含变量属性类型，缺失值单独分箱标志，分箱区间，对应woe值
        :return:
        """
        coef = dict(zip(model_feats, estimator.coef_[0]))
        intercept = estimator.intercept_[0]
        base_score = self.A - self.B * intercept

        res = {
            'name': ['base_score'],
            'attr_type': [np.nan],
            'null_flag': [np.nan],
            'bin': [None],
            'WOE': [np.nan],
            'score': [base_score]
        }
        for i, k in enumerate(bins_info):
            feat_type = bins_info[k]['type']
            feat_flag = bins_info[k]['flag']
            feat_bins = bins_info[k]['bins']
            feat_woes = bins_info[k]['woes']

            for j, v in enumerate(feat_bins):
                res['name'].append(k)
                res['attr_type'].append(feat_type)
                if feat_flag == 1 and j == 0:
                    res['null_flag'].append(1)
                else:
                    res['null_flag'].append(0)
                res['bin'].append(feat_bins[j])
                res['WOE'].append(feat_woes[j])
                res['score'].append(round(-self.B * feat_woes[j] * coef[k], 4))

        res = pd.DataFrame(res)

        return res
