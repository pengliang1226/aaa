# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:46
@file: FilterMethod.py
@desc: 过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
"""
from typing import Dict, List
import statsmodels.api as sm
import numpy as np
from pandas import DataFrame, Series
from statsmodels.stats.outliers_influence import variance_inflation_factor


def threshold_filter(X: DataFrame, null_threshold: float = 0.8, mode_threshold: float = 0.9,
                     unique_threshold: float = 0.8, null_flag: Dict = None) -> List:
    """
    根据缺失值比例、同值占比、唯一值占比筛选
    :param X: 数据
    :param null_threshold: 缺失值占比阈值
    :param mode_threshold: 同值占比阈值
    :param unique_threshold: 唯一值占比阈值
    :param null_flag: 缺失值标识符, list可能存在多个缺失值
    :return: 返回保留的变量
    """
    res = []
    for col in X.columns:
        col_data = X[col]
        if null_flag is not None:
            null_value = null_flag.get(col)
            col_data.replace(null_value, [np.nan] * len(null_value), inplace=True)

        # 缺失值判断
        null_rate = col_data.isna().sum() / col_data.shape[0]
        if null_rate > null_threshold:
            continue

        # 同值占比判断
        mode = col_data.mode()[0]
        mode_rate = (col_data == mode).sum() / col_data.dropna().shape[0]
        if mode_rate > mode_threshold:
            continue

        # 唯一值占比判断
        unique_rate = col_data.dropna().unique().size / col_data.dropna().shape[0]
        if unique_rate > unique_threshold:
            continue

        res.append(col)

    return res


def PSI_filter(df1: DataFrame, df2: DataFrame, bins_info: Dict, feature_type: Dict, threshold: float = 0.25) -> Dict:
    """
    psi筛选变量
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


def correlation_filter(df: DataFrame, col_list: List, threshold: float = 0.65) -> List:
    """
    相关性变量筛选, 默认选用Pearson, 字符型特征先进行转码
    :param df: 数据
    :param col_list: 变量列表，按iv值从大到小排序
    :param threshold: 相关系数阈值
    :return:
    """
    data_array = np.array(df[col_list])
    corr_result = np.corrcoef(data_array.T)

    res = []
    for i, col in enumerate(col_list):
        if i == 0:
            res.append(col)
        else:
            corr = corr_result[i, :i]
            if (corr < threshold).all():
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


def logit_pvalue_forward_filter(df: DataFrame, y: Series, col_list: List, p_value: float = 0.05) -> List:
    """
    logit显著性筛选, 前向逐步回归
    :param df: 数据
    :param y: y标签数据
    :param col_list: 变量列表
    :param p_value: p值
    :return:
    """
    pvalues_col = []
    # 按IV值逐个引入模型
    for col in col_list:
        pvalues_col.append(col)
        # 每引入一个特征就做一次显著性检验
        x_const = sm.add_constant(df.loc[:, pvalues_col])
        sm_lr = sm.Logit(y, x_const)
        sm_lr = sm_lr.fit()
        pvalue = sm_lr.pvalues[col]
        # 当引入的特征P值>=0.05时，则剔除，原先满足显著性检验的则保留，不再剔除
        if pvalue >= 0.05:
            pvalues_col.remove(col)

    return pvalues_col


# def logit_pvalue_backward_filter(df: DataFrame, y: Series, col_list: List, p_value: float = 0.05) -> List:
#     """
#     logit显著性筛选, 后向逐步回归
#     :param df: 数据
#     :param y: y标签数据
#     :param col_list: 变量列表
#     :param p_value: p值
#     :return:
#     """
#     x_c = x.copy()
#     # 所有特征引入模型，做显著性检验
#     x_const = sm.add_constant(x_c)
#     sm_lr = sm.Logit(y, x_const).fit()
#     pvalue_tup = [(i, j) for i, j in zip(sm_lr.pvalues.index, sm_lr.pvalues.values)][1:]
#     delete_count = len([i for i, j in pvalue_tup if j >= 0.05])
#     # 当有P值>=0.05的特征时，执行循环
#     while delete_count > 0:
#         # 按IV值从小到大的顺序依次逐个剔除
#         remove_col = [i for i, j in pvalue_tup if j >= 0.05][-1]
#         del x_c[remove_col]
#         # 每次剔除特征后都要重新做显著性检验，直到入模的特征P值都小于0.05
#         x2_const = sm.add_constant(x_c)
#         sm_lr2 = sm.Logit(y, x2_const).fit()
#         pvalue_tup2 = [(i, j) for i, j in zip(sm_lr2.pvalues.index, sm_lr2.pvalues.values)][1:]
#         delete_count = len([i for i, j in pvalue_tup2 if j >= 0.05])
#
#     pvalues_col = x_c.columns.tolist()
#
#     return pvalues_col