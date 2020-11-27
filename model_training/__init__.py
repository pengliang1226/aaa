# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/17 10:20
@file: __init__.py.py
@desc: 模型训练模块
"""
from model_training.BasicTrainer import BasicTrainer
from model_training.MergeTrainer import VotingTrainer, StackingTrainer, StackingCVTrainer
from model_training.RITrainer import HardCutoff, FuzzyAugmentation, ReWeighting, Extrapolation, \
    IterativeReclassification

__all__ = [
    'BasicTrainer',  # 基础模型
    # 模型融合
    'VotingTrainer',  # 投票法
    'StackingTrainer',  # stack法
    'StackingCVTrainer',  # stack法增加交叉验证
    # 拒绝推断
    'HardCutoff',  # 硬截断法（简单展开法）
    'FuzzyAugmentation',  # 模糊展开法
    'ReWeighting',  # 重新加权法
    'Extrapolation',  # 外推法
    'IterativeReclassification',  # 迭代再分类法
]
