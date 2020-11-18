# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/16 11:46
@file: BasicMethod.py
@desc: 基础性筛选方法、评分卡独有的方法等
"""
from typing import List, Dict

import numpy as np
import statsmodels.api as sm
from pandas import DataFrame, Series
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


def missing_filter(df: DataFrame, col_list: List, threshold: float = 0.75, null_flag: Dict = None) -> List:
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


def missingByMonth_filter(df: DataFrame, month: Series, col_list: List, threshold: float = 0.15,
                          null_flag: Dict = None) -> List:
    """
    缺失率按月筛选，按月计算变量缺失率，并计算相关指标查看波动情况，进行变量筛选
    [注]: 样本数据最好是连续月份，并且每月数据量不能差别过大
    :param df:
    :param month: 时间列, 只包含年月或者包含月份
    :param col_list: 变量列表
    :param threshold: 变异系数阈值，std/mean
    :param null_flag: 缺失值标识符
    :return:
    """
    data = df[col_list].copy()
    if null_flag is not None:
        for col in col_list:
            null_value = null_flag.get(col)
            if null_value is not None:
                data[col].replace(null_value, [np.nan] * len(null_value), inplace=True)

    data.loc[:, 'month'] = month
    group = data.groupby('month').apply(lambda x: x.isna().sum()).T.iloc[:-1]
    miss_num = group.apply(lambda x: x.sum(), axis=1)
    mean = group.apply(lambda x: x.mean(), axis=1)
    std = group.apply(lambda x: x.std(), axis=1)
    cv = std / mean
    res = [col for col in col_list if cv[col] < threshold or miss_num[col] < 100]  # 缺失值较少的列直接保留

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


def correlation_filter(df: DataFrame, col_list: List, threshold: float = 0.65, method='pearson') -> List:
    """
    变量间相关性变量筛选, 默认选用Pearson
    [注]:数据不能存在空值;
        pearson相关系数对异常值比较敏感，只对线性关系敏感，如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0;
        两个定序测量数据（顺序变量）之间也用spearman相关系数，不能用pearson相关系数;
    :param df: 数据
    :param col_list: 变量列表，按iv值从大到小排序
    :param threshold: 相关系数阈值
    :param method: 计算相关系数方法
    :return:
    """
    if df[col_list].isna().any().any():
        raise Exception('变量数据存在空值')

    data_array = np.array(df[col_list])
    if method == 'pearson':
        corr_result = np.fabs(np.corrcoef(data_array.T))
    elif method == 'spearman':
        corr_result = np.fabs(spearmanr(data_array)[0])
    else:
        raise Exception('未知的相关系数计算方式')

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
    if df[col_list].isna().any().any():
        raise Exception('变量数据存在空值')

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
        # 当引入特征后，若存在变量系数为负，则将引入的特征其剔除
        if (sk_lr.coef_[0] < 0).any():
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
        if (sk_lr.coef_[0] < 0).any():
            idx = np.where(sk_lr.coef_[0] < 0)[0][-1]
            col_list.remove(col_list[idx])
        else:
            break

    return col_list
