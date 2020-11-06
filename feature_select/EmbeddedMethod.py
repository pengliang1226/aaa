# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/26 11:44
@file: EmbeddedMethod.py
@desc: 嵌入法，实际上是学习器自主选择特征。得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。
"""
from typing import List, Union, Any

import numpy as np
from pandas import DataFrame, Series
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel


def Lasso_filter(df: DataFrame, y: Series, col_list: List, k: Union[int, float] = None, alpha: float = 1.0) -> List:
    """
    lasso特征筛选
    [注]: 需要考虑归一化或标准化问题
    :param df:
    :param y:
    :param col_list:
    :param k: 保留特征数目或比例, 当取值为None时, 默认取系数不为0的变量
    :param alpha:
    :return:
    """
    if df[col_list].isna().any():
        raise Exception('变量数据存在空值')
    if k >= 1 and isinstance(k, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(k, float):
        k = np.ceil(len(col_list) * k)

    if k is None:
        selector = SelectFromModel(Lasso(alpha=alpha, random_state=123))
    else:
        selector = SelectFromModel(Lasso(alpha=alpha, random_state=123), max_features=k, threshold=-np.inf)
    selector.fit(df[col_list], y)
    mask = selector.get_support()
    res = np.array(col_list)[mask].tolist()

    return res


def LassoCV_filter(df: DataFrame, y: Series, col_list: List, k: Union[int, float] = None, alphas: List = None) -> List:
    """
    lasso特征筛选（交叉验证）
    [注]: 需要考虑归一化或标准化问题
    :param df:
    :param y:
    :param col_list:
    :param k: 保留特征数目或比例, 当取值为None时, 默认取系数不为0的变量
    :param alphas: 交叉验证是需要的alpha列表
    :return:
    """
    if df[col_list].isna().any():
        raise Exception('变量数据存在空值')
    if k >= 1 and isinstance(k, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(k, float):
        k = np.ceil(len(col_list) * k)

    if k is None:
        selector = SelectFromModel(LassoCV(alphas=alphas, random_state=123))
    else:
        selector = SelectFromModel(LassoCV(alphas=alphas, random_state=123), max_features=k, threshold=-np.inf)
    selector.fit(df[col_list], y)
    mask = selector.get_support()
    res = np.array(col_list)[mask].tolist()

    return res


def model_filter(df: DataFrame, y: Series, col_list: List, estimator: Any, k: Union[int, float] = None) -> List:
    """
    树模型或线性模型筛选
    [注]: 线性模型需要考虑归一化或标准化问题；
         采用模型必须含有coef_或feature_importance参数；
         当k为None时，线性模型返回所有系数大于1*e-5特征，树模型返回特征重要性大于均值的特征
    :param df:
    :param y:
    :param col_list:
    :param estimator:
    :param k: 保留特征数目或比例, 当取值为None时, 默认取系数不为0的变量
    :return:
    """
    if df[col_list].isna().any():
        raise Exception('变量数据存在空值')
    if k >= 1 and isinstance(k, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(k, float):
        k = np.ceil(len(col_list) * k)

    if k is None:
        selector = SelectFromModel(estimator)
    else:
        selector = SelectFromModel(estimator, max_features=k, threshold=-np.inf)
    selector.fit(df[col_list], y)
    mask = selector.get_support()
    res = np.array(col_list)[mask].tolist()

    return res



