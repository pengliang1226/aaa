# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/29 10:04
@file: unuseless_util.py
@desc: 
"""
from typing import Any, Union, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series


# with open('bins.json', 'w') as f:
#     json.dump(QT.features_bins, f)
#
# with open('woe.json', 'w') as f:
#     json.dump(QT.features_woes, f)
#
# with open('iv.json', 'w') as f:
#     json.dump(QT.features_iv, f)
def get_time_span(col_time: Series, range_num: int = 10):
    """
    等分时间返回阈值
    :param col_time:
    :param range_num:
    :return:
    """
    col = pd.to_datetime(col_time)
    bins = pd.date_range(start=col.min(), end=col.max(), periods=range_num + 1, normalize=True)
    return bins


def get_time_span_by_month(col_time: Series) -> ndarray:
    """
    按月统计时间
    :param col_time:
    :return:
    """
    col = pd.to_datetime(col_time)
    bins = np.unique([str(x)[:7] for x in np.sort(col[col.notna()].unique())])
    return bins


def statistics_sample(X: Series, y: Series, bins: Any, bad_y: Any):
    """
    按时间统计样本
    :param X: 时间列
    :param y: 标签列
    :param bins: 时间划分阈值
    :param bad_y: 坏样本值
    :return:
    """
    X = pd.to_datetime(X)
    X = pd.cut(X, bins=bins, right=True, include_lowest=True, labels=False)

    time_span = []
    rate_bins = []
    bad_bins = []
    for v in range(len(bins) - 1):
        time_span.append('(' + str(bins[v])[:10] + ',' + str(bins[v + 1])[:10] + ']')
        if (X == v).sum() == 0:
            bad_bins.append('0')
            rate_bins.append('0')
        else:
            bad_rate = ((X == v) & (y == bad_y)).sum() / (X == v).sum()
            rate = (X == v).sum() / len(X)
            bad_bins.append(str(round(bad_rate, 4)))
            rate_bins.append(str(round(rate, 4)))
    if X.isna().any():
        time_span.append('NULL')
        bad_rate = ((X.isna()) & (y == bad_y)).sum() / (X.isna()).sum()
        rate = (X.isna()).sum() / len(X)
        bad_bins.append(str(round(bad_rate, 4)))
        rate_bins.append(str(round(rate, 4)))

    res = {
        'index': [str(i) for i in range(len(time_span))],
        'time_span': time_span,
        'bad_rate': bad_bins,
        'count_rate': rate_bins
    }
    return res


def statistics_sample_by_month(X: Series, y: Series, bins: Any, bad_y: Any):
    """
    按月统计样本
    :param X: 时间列
    :param y: 标签列
    :param bins: 月份列表
    :param bad_y: 坏样本值
    :return:
    """
    X = pd.to_datetime(X).apply(lambda x: str(x)[:7] if pd.notna(x) else None)

    time_span = []
    rate_bins = []
    bad_bins = []
    for v in bins:
        time_span.append(v)
        if (X == v).sum() == 0:
            bad_bins.append('0')
            rate_bins.append('0')
        else:
            bad_rate = ((X == v) & (y == bad_y)).sum() / (X == v).sum()
            rate = (X == v).sum() / len(X)
            bad_bins.append(str(round(bad_rate, 4)))
            rate_bins.append(str(round(rate, 4)))
    if X.isna().any():
        time_span.append('NULL')
        bad_rate = ((X.isna()) & (y == bad_y)).sum() / (X.isna()).sum()
        rate = (X.isna()).sum() / len(X)
        bad_bins.append(str(round(bad_rate, 4)))
        rate_bins.append(str(round(rate, 4)))

    res = {
        'index': [str(i) for i in range(len(time_span))],
        'time_span': time_span,
        'bad_rate': bad_bins,
        'count_rate': rate_bins
    }
    return res


def stat_group(X: Union[Series, ndarray], y: Union[Series, ndarray], tag_values: Any, bad_y: Any = 1):
    """
    分层统计
    :param X: 统计目标列
    :param y: 标签列
    :param tag_values: 需要统计的值列表
    :param bad_y: 坏样本值
    :return:
    """
    if isinstance(X, Series):
        X = X.values
    if isinstance(y, Series):
        y = y.values

    row_count = X.shape[0]
    res_map = {}
    for tag_val in tag_values:
        if (X == tag_val).sum() == 0:
            val_rate = 0
            bad_rate = 0
        else:
            val_rate = (X == tag_val).sum() / row_count
            bad_rate = ((X == tag_val) & (y == bad_y)).sum() / (X == tag_val).sum()
        tag_map = {'count_rate': str(round(val_rate, 4)), 'bad_rate': str(round(bad_rate, 4))}
        res_map[tag_val] = tag_map
    return res_map


def bins_to_str(bins: List, right: bool = True, is_num: bool = True) -> List:
    """
    将分箱阈值转换为字符串，以便输出
    :param bins: 分箱阈值
    :param right:
    :param is_num:
    :return:
    """
    res = []
    if is_num:
        for i in range(1, len(bins)):
            if right:
                temp = "({}, {}]".format(bins[i - 1], bins[i])
            else:
                temp = "[{}, {})".format(bins[i - 1], bins[i])
            res.append(temp)
    else:
        for v in bins:
            temp = "[{}]".format(",".join([str(i) for i in v]))
            res.append(temp)
    return res
