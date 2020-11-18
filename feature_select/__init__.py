# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:02
@file: __init__.py.py
@desc: 特征选择
"""
from feature_select.BasicMethod import dtype_filter, missing_filter, mode_filter, unique_filter, PSI_filter, \
    correlation_filter, vif_filter, logit_pvalue_forward_filter, logit_pvalue_backward_filter, coef_backward_filter, \
    coef_forward_filter, missingByMonth_filter
from feature_select.EmbeddedMethod import Lasso_filter, LassoCV_filter, model_filter
from feature_select.FilterMethod import variance_filter, corrY_filter, Chi2_filter, MI_filter, fclassif_filter
from feature_select.WrapperMethod import RFE_filter, RFECV_filter

__all__ = [
    # 基础方法和评分卡模型使用的方法
    'dtype_filter',  # 根据变量数据类型筛选
    'missing_filter',  # 根据变量缺失率筛选
    'missingByMonth_filter',  # 按月分析变量缺失值波动情况
    'mode_filter',  # 根据变量同值占比阈值筛选变量
    'unique_filter',  # 根据变量唯一值占比阈值筛选
    'PSI_filter',  # 根据变量psi筛选
    'correlation_filter',  # 变量间相关系数筛选
    'vif_filter',  # 变量间多重共线性筛选
    'logit_pvalue_forward_filter',  # 显著性筛选，向前回归
    'logit_pvalue_forward_filter',  # 显著性筛选，向后回归
    'coef_forward_filter',  # 逻辑回归系数一致性筛选，前向筛选
    'coef_backward_filter',  # 逻辑回归系数一致性筛选，后向筛选
    # 过滤法
    'variance_filter',  # 方差筛选
    'corrY_filter',  # 变量与y的相关性
    'Chi2_filter',  # 卡方检验
    'MI_filter',  # 互信息法
    'fclassif_filter',  # 方差分析筛选
    # 包装法
    'RFE_filter',  # 递归特征消除
    'RFECV_filter',  # 递归特征消除（交叉验证）
    # 嵌入法
    'Lasso_filter',  #
    'LassoCV_filter',  #
    'model_filter',  # 线性或树模型筛选
]
