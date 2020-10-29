# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:02
@file: __init__.py.py
@desc: 特征选择
"""
from feature_select.FilterMethod import threshold_filter, PSI_filter, correlation_filter, vif_filter

__all__ = [
    'threshold_filter',  # 根据缺失值，同值占比，唯一值占比阈值筛选变量
    'PSI_filter',  # 计算psi筛选变量
    'correlation_filter',  # 相关系数筛选
    'vif_filter',  # 多重共线性筛选
]
