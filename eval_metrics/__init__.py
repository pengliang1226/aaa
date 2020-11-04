# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/30 10:20
@file: __init__.py.py
@desc: 模型评价指标
"""
from eval_metrics.BinaryClassifier import *
from eval_metrics.ScoreStretch import ScoreStretch
from eval_metrics.StatisticalReport import calc_score_distribution, calc_distribute_report


__all__ = [
    # 模型转换为评分
    'ScoreStretch',  # 概率转换为评分
    # 分数分布统计
    'calc_score_distribution',  # 模型评分分布
    'calc_distribute_report',  # 模型评分整体分布情况报表
    # 二分类
    'calc_f1',
    'calc_ks',
    'calc_acc',
    'calc_gain',
    'calc_gini',
    'calc_lift',
    'calc_psi',
    'calc_roc',
    'calc_recall',
    'calc_precision',
    'calc_auc',
    # 多分类
    # 回归
]
