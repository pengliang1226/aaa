# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:02
@file: __init__.py.py
@desc: 特征选择
"""
from feature_select.FilterMethod import correlation_filter, vif_filter
from feature_select._base import dtype_filter, nan_filter, mode_filter, unique_filter, PSI_filter, \
    logit_pvalue_forward_filter, logit_pvalue_backward_filter, coef_backward_filter, coef_forward_filter

__all__ = [
    'nan_filter',  # 根据缺失值阈值筛选变量
    'mode_filter',  # 根据同值占比阈值筛选变量
    'unique_filter',  # 根据唯一值占比阈值筛选变量
    'PSI_filter',  # 计算psi筛选变量
    'correlation_filter',  # 相关系数筛选
    'vif_filter',  # 多重共线性筛选
    'logit_pvalue_forward_filter',  # 显著性筛选，向前回归
    'logit_pvalue_forward_filter',  # 显著性筛选，向后回归
    'coef_forward_filter',  # 逻辑回归系数一致性筛选，前向筛选
    'coef_backward_filter',  # 逻辑回归系数一致性筛选，后向筛选
]
