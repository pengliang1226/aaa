# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/27 11:10
@file: __init__.py.py
@desc:
"""
from util.calc_util import woe_single_all, get_attr_by_dtype, get_attr_by_unique, divide_sample, disorder_mapping, \
    woe_single, woe_transform
from util.config import CONTINUOUS, DISCRETE

__all__ = [
    'woe_single',  # 单个计算woe
    'woe_single_all',  # 批量计算woe
    'woe_transform',  # woe编码转换
    'get_attr_by_dtype',  # 根据数据类型定义变量类型
    'get_attr_by_unique',  # 根据唯一值个数定义变量类型
    'divide_sample',  # 切分数据
    'disorder_mapping',  # 无序变量转码
    'CONTINUOUS',  # 定量变量标识
    'DISCRETE',  # 定性变量标识
]
