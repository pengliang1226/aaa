# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:46
@file: _base.py
@desc: 基础性筛选方法、评分卡独有的方法等
"""
from typing import List, Dict

import numpy as np
import statsmodels.api as sm
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression


def dtype_filter(df: DataFrame, col_list: List) -> List:
    """
    根据数据类型筛选，剔除掉字符型变量; 后期可以增加剔除变量类型
    :param df:
    :param col_list: 变量列表
    :return:
    """
    res = []
    for col in col_list:
        if df[col].dtype != 'O':
            res.append(col)
    return res


def nan_filter(df: DataFrame, col_list: List, threshold: float = 0.75, null_flag: Dict = None) -> List:
    """
    缺失率筛选
    :param df:
    :param col_list: 变量列表
    :param threshold: 缺失值阈值
    :param null_flag: 缺失值标识符
    :return:
    """
    res = []
    for col in col_list:
        col_data = df[col]
        if null_flag is not None and null_flag.get(col) is not None:
            null_value = null_flag.get(col)
            null_rate = (col_data.isna() | col_data.isin(null_value)).sum() / col_data.shape[0]
        else:
            null_rate = col_data.isna() / col_data.shape[0]

        if null_rate <= threshold:
            res.append(col)
    return res


def mode_filter(df: DataFrame, col_list: List, threshold: float = 0.9, null_flag: Dict = None) -> List:
    """
    众数占比筛选
    :param df:
    :param col_list: 变量列表
    :param threshold:
    :param null_flag:
    :return:
    """
    res = []
    for col in col_list:
        col_data = df[col]
        if null_flag is not None and null_flag.get(col) is not None:
            null_value = null_flag.get(col)
            col_data = col_data[~(col_data.isna() | col_data.isin(null_value))]
        else:
            col_data = col_data[~col_data.isna()]

        mode_rate = col_data.value_counts().iloc[0] / col_data.shape[0]
        if mode_rate <= threshold:
            res.append(col)
    return res


def unique_filter(df: DataFrame, col_list: List, threshold: float = 0.9, null_flag: Dict = None) -> List:
    """
    唯一值占比筛选
    :param df:
    :param col_list: 变量列表
    :param threshold:
    :param null_flag:
    :return:
    """
    res = []
    for col in col_list:
        col_data = df[col]
        if null_flag is not None and null_flag.get(col) is not None:
            null_value = null_flag.get(col)
            col_data = col_data[~(col_data.isna() | col_data.isin(null_value))]
        else:
            col_data = col_data[~col_data.isna()]

        unique_rate = col_data.unique().size / col_data.shape[0]
        if unique_rate <= threshold:
            res.append(col)
    return res


def PSI_filter(df1: DataFrame, df2: DataFrame, bins_info: Dict, feature_type: Dict, threshold: float = 0.25) -> Dict:
    """
    psi筛选变量, psi: [0-0.1] 好；[0.1-0.25] 略不稳定；[0.25-] 不稳定
    :param df1:
    :param df2:
    :param bins_info: 每个变量分箱信息
    :param feature_type: 每个变量属性类型
    :param threshold: psi阈值
    :return: 保留变量Dict
    """
    res = {}
    N1 = df1.shape[0]
    N2 = df2.shape[0]

    for col in bins_info:
        col_data1 = df1[col]
        col_data2 = df2[col]
        bins = bins_info[col]['bins']  # 分箱阈值
        flag = bins_info[col]['flag']  # 缺失值是否单独一箱
        attr_type = feature_type[col]

        # 存储各分箱区间数量占比
        rate_bins1 = []
        rate_bins2 = []

        if flag == 1:
            mask1 = col_data1.isin(bins[0])
            mask2 = col_data2.isin(bins[0])
            rate_bins1.append(mask1.sum() / N1)
            rate_bins2.append(mask2.sum() / N2)
            bins = bins[1:]
            col_data1 = col_data1[~mask1]
            col_data2 = col_data2[~mask2]

        if attr_type == 1:
            for left, right in bins:
                mask1 = (col_data1 > left) & (col_data1 <= right)
                mask2 = (col_data2 > left) & (col_data2 <= right)
                rate_bins1.append(mask1.sum() / N1)
                rate_bins2.append(mask2.sum() / N2)
        else:
            for v in bins:
                mask1 = col_data1.isin(v)
                mask2 = col_data2.isin(v)
                rate_bins1.append(mask1.sum() / N1)
                rate_bins2.append(mask2.sum() / N2)

        rate_bins1 = np.array(rate_bins1)
        rate_bins2 = np.array(rate_bins2)
        psi = round(np.sum((rate_bins2 - rate_bins1) * np.log(rate_bins2 / rate_bins1)), 4)
        if psi < threshold:
            res[col] = psi

    return res


def logit_pvalue_forward_filter(df: DataFrame, y: Series, col_list: List, threshold: float = 0.05) -> List:
    """
    logit显著性筛选, 前向逐步回归
    :param df: 数据
    :param y: y标签数据
    :param col_list: 变量列表, 按iv从大到小排序
    :param threshold: p值阈值
    :return:
    """
    pvalues_col = []
    # 按IV值逐个引入模型
    for col in col_list:
        pvalues_col.append(col)
        # 每引入一个特征就做一次显著性检验
        x_const = sm.add_constant(df.loc[:, pvalues_col])
        sm_lr = sm.Logit(y, x_const).fit(disp=False)
        pvalue = sm_lr.pvalues[col]
        # 当引入的特征P值>=0.05时，则剔除，原先满足显著性检验的则保留，不再剔除
        if pvalue >= threshold:
            pvalues_col.remove(col)

    return pvalues_col


def logit_pvalue_backward_filter(df: DataFrame, y: Series, col_list: List, threshold: float = 0.05) -> List:
    """
    logit显著性筛选, 后向逐步回归
    :param df: 数据
    :param y: y标签数据
    :param col_list: 变量列表, 按iv从大到小排序
    :param threshold: p值
    :return:
    """
    x_c = df.loc[:, col_list].copy()
    # 所有特征引入模型，做显著性检验
    x_const = sm.add_constant(x_c)
    sm_lr = sm.Logit(y, x_const).fit()
    delete_count = np.where(sm_lr.pvalues >= threshold)[0].size
    # 当有P值>=0.05的特征时，执行循环
    while delete_count > 0:
        # 按IV值从小到大的顺序依次逐个剔除
        remove_col = sm_lr.pvalues.index[np.where(sm_lr.pvalues >= threshold)[0][-1]]
        del x_c[remove_col]
        # 每次剔除特征后都要重新做显著性检验，直到入模的特征P值都小于0.05
        x_const = sm.add_constant(x_c)
        sm_lr = sm.Logit(y, x_const).fit()
        delete_count = np.where(sm_lr.pvalues >= threshold)[0].size

    pvalues_col = x_c.columns.tolist()

    return pvalues_col


def coef_forward_filter(df: DataFrame, y: Series, col_list: List) -> List:
    """
    系数一致筛选, 保证逻辑回归系数全为正, 前向筛选
    :param df: 数据
    :param y: y标签数据
    :param col_list: 变量列表, 按iv从大到小排序
    :return:
    """
    coef_col = []
    # 按IV值逐个引入模型，输出系数
    for i, col in enumerate(col_list):
        coef_col.append(col)
        X = df.loc[:, coef_col]
        sk_lr = LogisticRegression(random_state=0).fit(X, y)
        coef_dict = {k: v for k, v in zip(coef_col, sk_lr.coef_[0])}
        # 当引入特征的系数为负，则将其剔除
        if coef_dict[col] < 0:
            coef_col.remove(col)

    return coef_col


def coef_backward_filter(df: DataFrame, y: Series, col_list: List) -> List:
    """
    系数一致筛选, 保证逻辑回归系数全为正, 后向筛选
    :param df: 数据
    :param y: y标签数据
    :param col_list: 变量列表, 按iv从大到小排序
    :return:
    """
    # 按IV值逐个引入模型，输出系数
    while True:
        X = df.loc[:, col_list]
        sk_lr = LogisticRegression(random_state=0).fit(X, y)
        if any(sk_lr.coef_[0] < 0):
            idx = np.where(sk_lr.coef_[0] < 0)[0][-1]
            col_list.remove(col_list[idx])
        else:
            break

    return col_list
