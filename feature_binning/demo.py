# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2021/1/18 15:53
@file: demo.py
@desc: 
"""
from math import inf

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

__SMOOTH__ = 1e-6
__DEFAULT__ = 1e-6


def woe_single(B: float, G: float, b, g):
    """
    woe计算
    """
    res = np.log(((b + __SMOOTH__) / (B + __SMOOTH__)) / ((g + __SMOOTH__) / (G + __SMOOTH__)))
    res = np.where(np.isinf(res), __DEFAULT__, res)
    return res


def encode_woe(X, y):
    """
    定性变量woe有序转码，根据woe从小到大排序，替换为0-n数字, 返回转码后对应关系
    :param X: 变量数据
    :param y: y标签数据
    :return: 返回转码后的Dict
    """
    B = y.sum()
    G = y.size - B
    unique_value = X.unique()
    mask = (unique_value.reshape(-1, 1) == X.values)
    mask_bad = mask & (y.values == 1)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    woe_value = woe_single(B, G, b, g)
    woe_value_sort = np.argsort(woe_value)
    res = dict(zip(unique_value, woe_value_sort))
    return res


def DTBinning(X, y, max_leaf_nodes=5, min_samples_leaf=0.05):
    """
    获取决策树分箱结果
    :param X: 单个变量数据
    :param y: 标签数据
    :param max_leaf_nodes: 分箱数量
    :param min_samples_leaf: 每个分箱最小样本量占比
    :return: 决策树分箱区间
    """
    # 初步分箱
    if X.unique().size <= max_leaf_nodes:  # 如果变量唯一值个数小于分箱数, 则直接按唯一值作为阈值
        cutoffs = [-inf]
        cutoffs.extend(np.sort(X.unique()))
        cutoffs = np.array(cutoffs)
    else:
        tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                      criterion='gini', max_depth=None, min_samples_split=2, random_state=1234)
        tree.fit(X.values.reshape(-1, 1), y)
        cutoffs = [-inf]
        cutoffs.extend(np.sort(tree.tree_.threshold[tree.tree_.feature != -2]))
        cutoffs.append(inf)
        cutoffs = np.array(cutoffs)

    # 分箱中只存在好或坏客户的箱体
    # 首先判断首个区间的方向
    # 如果b/g大于B/G则区间方向定为b/g减小的方向, 如果区间全是好客户则向后合并, 反之向前;
    # 如果b/g大于B/G则区间方向定为b/g增大的方向, 如果区间全是好客户则向前合并, 反之向后;
    B_G_rate = y.sum() / (y.size - y.sum())
    x_cut = pd.cut(X, cutoffs, include_lowest=True, labels=False)
    cutoffs = cutoffs[1:]
    freq_tab = pd.crosstab(x_cut, y)
    freq = freq_tab.values
    value_counts = freq[:, 1] / freq[:, 0]

    while np.where(freq == 0)[0].size > 0:
        idx = np.where(freq == 0)[0][0]
        if idx == 0:
            tmp = freq[idx] + freq[idx + 1]
            freq[idx + 1] = tmp
            # 删除对应的切分点
            cutoffs = np.delete(cutoffs, idx)
        elif idx == len(cutoffs) - 1:
            tmp = freq[idx - 1] + freq[idx]
            freq[idx - 1] = tmp
            # 删除对应的切分点
            cutoffs = np.delete(cutoffs, idx - 1)
        else:
            if value_counts[0] >= B_G_rate:  # 方向定为b/g减小的方向
                if np.where(freq[idx] > 0)[0] == 0:  # 区间全为好样本
                    tmp = freq[idx] + freq[idx + 1]
                    freq[idx + 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx)
                else:
                    tmp = freq[idx - 1] + freq[idx]
                    freq[idx - 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx - 1)
            else:
                if np.where(freq[idx] > 0)[0] == 0:  # 区间全为好样本
                    tmp = freq[idx - 1] + freq[idx]
                    freq[idx - 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx - 1)
                else:
                    tmp = freq[idx] + freq[idx + 1]
                    freq[idx + 1] = tmp
                    # 删除对应的切分点
                    cutoffs = np.delete(cutoffs, idx)

        # 删除idx
        freq = np.delete(freq, idx, 0)
        # 更新value_counts
        value_counts = freq[:, 1] / freq[:, 0]

    threshold = [-inf]
    threshold.extend(cutoffs[:-1].tolist())
    threshold.append(inf)

    return threshold


def calc_iv(data, flag_y='y', null_value=None, max_leaf_nodes=5, min_samples_leaf=0.05):
    """
    计算变量iv值
    :param data: 数据
    :param flag_y: y标签列名
    :param null_value: 缺失值标识符
    :param max_leaf_nodes: 分箱数量
    :param min_samples_leaf: 每个分箱最小样本量占比
    :return:
    """
    data_y = data.pop(flag_y)
    B = data_y.sum()
    G = data_y.size - B
    bins_map = {}
    iv_map = {}

    for col in data.columns:
        b_bins = []
        g_bins = []
        flag = 0  # 标识缺失值是否单独做为一箱

        X = data[col].copy()
        y = data_y.copy()
        # 缺失值数量判断，超过5%单独一箱
        mask = (X.isin(null_value))
        null_rate = mask.sum() / X.size
        if null_rate >= 0.05:
            b_bins.append(y[mask].sum())
            g_bins.append(mask.sum() - y[mask].sum())
            y = y[~mask]
            X = X[~mask]
            flag = 1

        if X.dtype != 'O':
            bucket = DTBinning(X, y, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
            bucket = [[bucket[i], bucket[i + 1]] for i in range(len(bucket) - 1)]
            for left, right in bucket:
                mask = (X > left) & (X <= right)
                b_bins.append(y[mask].sum())
                g_bins.append(mask.sum() - y[mask].sum())
        else:
            bin_map = encode_woe(X, y)
            X_map = X.map(bin_map)
            bins = DTBinning(X_map, y, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
            keys = np.array(list(bin_map.keys()))
            values = np.array(list(bin_map.values()))
            bucket = []
            for i in range(len(bins) - 1):
                mask = (values > bins[i]) & (values <= bins[i + 1])
                bucket.append(keys[mask].tolist())

            for v in bins:
                mask = X.isin(v)
                b_bins.append(y[mask].sum())
                g_bins.append(mask.sum() - y[mask].sum())

        if flag == 1:
            bucket.insert(0, null_value)

        b_bins = np.array(b_bins)
        g_bins = np.array(g_bins)
        woes = woe_single(B, G, b_bins, g_bins).tolist()
        temp = (b_bins + __SMOOTH__) / (B + __SMOOTH__) - (g_bins + __SMOOTH__) / (G + __SMOOTH__)
        iv = float(np.around((temp * woes).sum(), 6))

        bins_map[col] = bucket
        iv_map[col] = iv

    return bins_map, iv_map


if __name__ == '__main__':
    data = pd.read_csv('test.csv')
    flag_y = 'CLASS'  # 修改为样本中y标签列的列名
    bad_y = 'Y'  # y标签列中坏样本对应的值
    null_value = [-999]  # 修改为数据中的缺失值标识符，可以多个用逗号隔开

    if bad_y != 1:
        data[flag_y] = data[flag_y].apply(lambda x: 1 if x == bad_y else 0)
    res = calc_iv(data.copy(), flag_y=flag_y, null_value=null_value)