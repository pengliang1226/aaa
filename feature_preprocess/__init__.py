# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/30 15:30
@file: __init__.py.py
@desc: 
"""
from feature_preprocess.Imputer import statistics_imputer, interpolate_imputer, knn_imputer
from feature_preprocess.Outlier import z_score_check, IQR_check
from feature_preprocess.Scaling import FeatureScaling

__all__ = [
    # 缺失值填充
    'statistics_imputer',
    'interpolate_imputer',
    'knn_imputer',
    # 异常值检测
    'z_score_check',
    'IQR_check',
    # 特征缩放
    'FeatureScaling',
]
