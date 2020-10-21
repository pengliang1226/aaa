# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/27 11:10
@file: __init__.py.py
@desc:
"""
from util.calc_util import woe_single_all, get_feature_type
from util.config import CONTINUOUS, DISCRETE


__all__ = [
    'woe_single_all',  # 计算woe
    'get_feature_type',  # 自动定义变量类型
    'CONTINUOUS',  # 定量变量标识
    'DISCRETE',  # 定性变量标识
]