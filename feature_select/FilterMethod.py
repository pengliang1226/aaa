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
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


def Chi2_filter(df: DataFrame, y: Series, col_list: List, keep: Union[int, float] = None) -> List:
    """
    卡方检验筛选：建议作为分类问题的分类变量的筛选方法, 检验定性自变量对定性因变量的相关性。
    [注]:数据不能存在负数和空值; 如果变量名称不是字符型，返回结果可能会改变为字符型报错
    :param df:
    :param y:
    :param col_list:
    :param keep: 保留特征数目或比例
    :return:
    """
    if df[col_list].isna().any() or (df[col_list] < 0).any():
        raise Exception('变量数据存在空值或负值')
    if keep < 0:
        raise Exception('参数keep需大于0')
    if keep >= 1 and isinstance(keep, float):
        raise Exception('参数keep大于等于1时, 请输入整数')
    chi_value, _ = chi2(df[col_list], y)
    idx = np.argsort(-chi_value)
    if isinstance(keep, float):
        keep = np.ceil(keep * len(col_list))
        res = np.array(col_list)[idx[:keep]].tolist()
    else:
        res = np.array(col_list)[idx[:keep]].tolist()

    return res


def MI_filter(df: DataFrame, y: Series, col_list: List, keep: Union[int, float] = None,
              discrete_index: List = None) -> List:
    """
    互信息(Mutual information)筛选
    [注]:数据不能存在空值, 针对分类场景；如果变量名称不是字符型，返回结果可能会改变为字符型报错
    :param df:
    :param y:
    :param col_list:
    :param keep: 保留特征数目或比例
    :param discrete_index: list离散特征在特征列表的索引
    :return:
    """
    if df[col_list].isna().any() or (df[col_list] < 0).any():
        raise Exception('变量数据存在空值或负值')
    if keep < 0:
        raise Exception('参数keep需大于0')
    if keep >= 1 and isinstance(keep, float):
        raise Exception('参数keep大于等于1时, 请输入整数')

    mi_value = mutual_info_classif(df[col_list], y,
                                   discrete_features='auto' if discrete_index is None else discrete_index,
                                   random_state=123)
    idx = np.argsort(-mi_value)
    if isinstance(keep, float):
        keep = np.ceil(keep * len(col_list))
        res = np.array(col_list)[idx[:keep]].tolist()
    else:
        res = np.array(col_list)[idx[:keep]].tolist()

    return res


def correlation_filter(df: DataFrame, col_list: List, threshold: float = 0.65) -> List:
    """
    变量间相关性变量筛选, 默认选用Pearson
    [注]: Pearson相关系数的一个明显缺陷是，作为特征排序机制，他只对线性关系敏感。
         如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0。
    :param df: 数据
    :param col_list: 变量列表，按iv值从大到小排序
    :param threshold: 相关系数阈值
    :return:
    """
    data_array = np.array(df[col_list])
    corr_result = np.fabs(np.corrcoef(data_array.T))

    idx = []
    res = []
    for i, col in enumerate(col_list):
        if i == 0:
            idx.append(i)
            res.append(col)
        else:
            corr = corr_result[i, idx]
            if (corr < threshold).all():
                idx.append(i)
                res.append(col)

    return res


def vif_filter(df: DataFrame, col_list: List, threshold=10) -> List:
    """
    多重共线性筛选，逐步剔除vif大于阈值的变量，直至所有变量的vif值小于阈值
    :param df: 数据
    :param col_list: 变量列表，按iv值从大到小排序
    :param threshold: vif阈值，大于该值表示存在多重共线性
    :return:
    """
    vif_array = np.array(df[col_list])
    vifs_list = [variance_inflation_factor(vif_array, i) for i in range(vif_array.shape[1])]
    vif_high = [k for k, v in zip(col_list, vifs_list) if v >= threshold]
    if len(vif_high) > 0:
        for col in reversed(vif_high):
            col_list.remove(col)
            vif_array = np.array(df[col_list])
            vifs_list = [variance_inflation_factor(vif_array, i) for i in range(vif_array.shape[1])]
            if (np.array(vifs_list) < threshold).all():
                break

    return col_list
