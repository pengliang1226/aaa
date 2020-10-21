# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/27 11:10
@file: __init__.py.py
@desc: 
"""
from feature_binning.ChiMergeBinner import ChiMergeBinner
from feature_binning.DecisionTreeBinner import DecisionTreeBinner
from feature_binning.QuantileBinner import QuantileBinner

# TODO y标签数据必须为0,1; 变量中不能既包含空值又包含缺失值标识符; 分箱区间默认左开右闭
__all__ = [
    'DecisionTreeBinner',  # 决策树分箱
    'ChiMergeBinner',  # 卡方分箱
    # 'QuantileBinner'  # 等频分箱
]
