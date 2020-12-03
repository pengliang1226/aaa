# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:46
@file: FilterMethod.py
@desc: 过滤法，设定阈值或者待选择特征的个数进行筛选，即设计一些统计量来过滤特征，并不考虑后续学习器问题。
"""
from typing import List, Union

import numpy as np
from pandas import DataFrame, Series
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif, f_classif, SelectKBest


def variance_filter(df: DataFrame, col_list: List, threshold: float = 0.5) -> List:
    """
    方差筛选, 建议作为数值特征的筛选方法。
    [使用说明]:只会使用阈值为0或者阈值很小的方差过滤，来为我们优先消除一些明显用不到的特征，然后我们会选择更优的特征选择方法
    [注]:可以存在空值，最好把缺失值标识符替换为空不然会影响结果；如果变量名称不是字符型，返回结果可能会改变为字符型报错
    :param df:
    :param col_list:
    :param threshold: 方差阈值
    :return:
    """
    selector = VarianceThreshold()
    selector.fit(df[col_list])
    mask = (selector.variances_ >= threshold)
    res = np.array(col_list)[mask].tolist()

    return res


def corrY_filter(df: DataFrame, y: Series, col_list: List, k: Union[int, float] = None,
                 method: str = 'pearson') -> List:
    """
    相关系数筛选：线性回归和逻辑回归的时候会使用pearson相关系数，而如果是树模型则更倾向于使用spearman相关系数。
    [注]:数据不能存在空值;
        pearson相关系数对异常值比较敏感，只对线性关系敏感，如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0;
        两个定序测量数据（顺序变量）之间用spearman相关系数，不能用pearson相关系数;
    :param df:
    :param y:
    :param col_list:
    :param k: 保留特征数目或比例
    :param method: 计算相关系数方法
    :return:
    """
    if df[col_list].isna().any().any():
        raise Exception('变量数据存在空值')
    if k >= 1 and isinstance(k, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(k, float):
        k = np.ceil(len(col_list) * k)

    def udf(X, y):
        if method == 'pearson':
            result = np.array([pearsonr(x, y) for x in X.T])
        elif method == 'spearman':
            result = np.array([spearmanr(x, y) for x in X.T])
        else:
            raise Exception('未知相关系数计算方法')
        return np.absolute(result[:, 0]), result[:, 1]

    selector = SelectKBest(udf, k=k)
    selector.fit(df[col_list], y)
    mask = selector.get_support()
    res = np.array(col_list)[mask].tolist()

    return res


def Chi2_filter(df: DataFrame, y: Series, col_list: List, k: Union[int, float] = None) -> List:
    """
    卡方检验筛选：建议作为分类问题的分类变量的筛选方法, 检验定性自变量对定性因变量的相关性。
    [注]:数据不能存在负数和空值
    :param df:
    :param y:
    :param col_list:
    :param k: 保留特征数目或比例
    :return:
    """
    if df[col_list].isna().any().any() or (df[col_list] < 0).any().any():
        raise Exception('变量数据存在空值或负值')
    if k >= 1 and isinstance(k, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(k, float):
        k = np.ceil(len(col_list) * k)

    selector = SelectKBest(chi2, k=k)
    selector.fit(df[col_list], y)
    mask = selector.get_support()
    res = np.array(col_list)[mask].tolist()

    return res


def MI_filter(df: DataFrame, y: Series, col_list: List, k: Union[int, float] = None,
              discrete_index: List = None) -> List:
    """
    互信息(Mutual information)筛选
    [注]:数据不能存在空值, 针对分类场景
    :param df:
    :param y:
    :param col_list:
    :param k: 保留特征数目或比例
    :param discrete_index: list离散特征在特征列表的索引
    :return:
    """
    if df[col_list].isna().any().any():
        raise Exception('变量数据存在空值')
    if k >= 1 and isinstance(k, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(k, float):
        k = np.ceil(len(col_list) * k)

    mi_value = mutual_info_classif(df[col_list], y,
                                   discrete_features='auto' if discrete_index is None else discrete_index,
                                   random_state=123)
    idx = np.argsort(-mi_value)
    res = np.array(col_list)[idx[:k]].tolist()

    return res


def fclassif_filter(df: DataFrame, y: Series, col_list: List, k: Union[int, float] = None) -> List:
    """
    ANOVA方差分析筛选
    [注]:数据不能存在空值
    :param df:
    :param y:
    :param col_list:
    :param k: 保留特征数目或比例
    :return:
    """
    if df[col_list].isna().any().any() or (df[col_list] < 0).any().any():
        raise Exception('变量数据存在空值或负值')
    if k >= 1 and isinstance(k, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    if isinstance(k, float):
        k = np.ceil(len(col_list) * k)

    selector = SelectKBest(f_classif, k=k)
    selector.fit(df[col_list], y)
    mask = selector.get_support()
    res = np.array(col_list)[mask].tolist()

    return res
