# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/9/28 18:21
@file: StatisticalReport.py
@desc: 统计模型报表
"""
from typing import Union, Dict

import numpy as np
from numpy import ndarray
from pandas import Series


def calc_score_distribution(score: ndarray, y: Union[Series, ndarray], min_: float = 300, max_: float = 1000) -> Dict:
    """计算模型评分分布"""
    if isinstance(y, Series):
        y = y.values

    count = []
    bad_rate = []
    interval = []
    n = int(np.ceil((max_ - min_) / 50))
    for i in range(n):
        left_interval = int(min_ + i * 50)
        if i == (n - 1):
            right_interval = int(max_ + 1)
            interval.append('[' + str(left_interval) + ',' + str(int(max_)) + ']')
        else:
            right_interval = int(min_ + (i + 1) * 50)
            interval.append('[' + str(left_interval) + ',' + str(right_interval) + ')')
        mask = (score >= left_interval) & (score < right_interval)
        count.append(mask.sum())
        bad_rate.append(y[mask].sum() / mask.sum() if mask.sum() > 0 else 0)

    res = {"count": [i for i in count],
           "bad_rate": [round(i, 4) for i in bad_rate] if y is not None else [],
           "interval": interval}

    return res


def calc_distribute_report(score: ndarray, y: ndarray, min_: float = 300, max_: float = 1000) -> Dict:
    """
    计算整体样本的分布报告
    :param score:
    :param y:
    :param max_:
    :param min_:
    :return:
    """
    res = {}
    res['number_interval'] = distribute_by_number(score, y)
    res['score_interval'] = distribute_by_score(score, y, max_=max_, min_=min_)
    return res


def distribute_by_number(score: ndarray, y: ndarray) -> dict:
    """
    按人数区间统计
    :param score:
    :param y:
    :return:
    """
    index = np.argsort(score)
    score = score[index]
    y = y[index]
    percentile_bins = ['[' + str(i) + ',' + str(i + 5) + ')' if i != 95 else '[95,100]' for i in range(0, 100, 5)]
    rate_bins = [0.05] * len(percentile_bins)
    threshold = [int(len(score) * i / 100) for i in range(0, 105, 5)]

    B = y.sum()
    G = len(y) - B
    bad_rate_bins = []  # 区间坏客户率
    total_bad_rate_bins = []  # 累计坏客户占比
    total_good_rate_bins = []  # 累计好客户占比
    ks_bins = []  # 好坏区分程度
    pass_bad_rate_bins = []  # 通过坏客户率
    refuse_bad_rate_bins = []  # 拒绝坏客户率
    pass_rate_bins = []  # 通过率

    for i in range(len(threshold) - 1):
        right = threshold[i + 1]
        left = threshold[i]

        bad_rate = round(y[left:right].sum() / y[left:right].size, 4)

        b = y[:right].sum()
        g = len(y[:right]) - b
        total_bad_rate = round(b / B, 4)
        total_good_rate = round(g / G, 4)
        ks = round(abs(total_bad_rate - total_good_rate), 4)

        pass_bad_rate = round(y[left:].sum() / len(y[left:]), 4)
        refuse_bad_rate = round(y[:right].sum() / len(y[:right]), 4)
        pass_rate = round(len(y[left:]) / len(y), 4)

        bad_rate_bins.append(bad_rate)
        total_bad_rate_bins.append(total_bad_rate)
        total_good_rate_bins.append(total_good_rate)
        ks_bins.append(ks)
        pass_bad_rate_bins.append(pass_bad_rate)
        refuse_bad_rate_bins.append(refuse_bad_rate)
        pass_rate_bins.append(pass_rate)

    res = {"percentile_bins": percentile_bins,
           "rate_bins": rate_bins,
           "bad_rate_bins": bad_rate_bins,
           "total_bad_rate_bins": total_bad_rate_bins,
           "total_good_rate_bins": total_good_rate_bins,
           "ks_bins": ks_bins,
           "pass_bad_rate_bins": pass_bad_rate_bins,
           "refuse_bad_rate_bins": refuse_bad_rate_bins,
           "pass_rate_bins": pass_rate_bins}

    return res


def distribute_by_score(score: ndarray, y: ndarray, max_=1000, min_=300) -> dict:
    """
    按分数区间统计
    :param score:
    :param y:
    :param max_:
    :param min_:
    :return:
    """
    index = np.argsort(score)
    score = score[index]
    y = y[index]

    B = y.sum()
    G = len(y) - B
    percentile_bins = []
    rate_bins = []
    bad_rate_bins = []  # 区间坏客户率
    total_bad_rate_bins = []  # 累计坏客户占比
    total_good_rate_bins = []  # 累计好客户占比
    ks_bins = []  # 好坏区分程度
    pass_bad_rate_bins = []  # 通过坏客户率
    refuse_bad_rate_bins = []  # 拒绝坏客户率
    pass_rate_bins = []  # 通过率

    n = int(np.ceil((max_ - min_) / 25))
    for i in range(n):
        left = int(min_ + i * 25)
        if i == (n - 1):
            right = int(max_ + 1)
            percentile_bins.append('[' + str(left) + ',' + str(int(max_)) + ']')
        else:
            right = int(min_ + (i + 1) * 25)
            percentile_bins.append('[' + str(left) + ',' + str(right) + ')')

        mask_up_down = (score >= left) & (score < right)
        rate = round(mask_up_down.sum() / score.size, 4)
        bad_rate = round(y[mask_up_down].sum() / mask_up_down.sum() if mask_up_down.any() else 0., 4)

        mask_up = score < right
        mask_down = score >= left
        b = y[mask_up].sum()
        g = mask_up.sum() - b
        total_bad_rate = round(b / B, 4)
        total_good_rate = round(g / G, 4)
        ks = round(abs(total_bad_rate - total_good_rate), 4)

        pass_bad_rate = round(y[mask_down].sum() / mask_down.sum() if mask_down.any() else 0., 4)
        refuse_bad_rate = round(y[mask_up].sum() / mask_up.sum() if mask_up.any() else 0., 4)
        pass_rate = round(mask_down.sum() / len(y), 4)

        rate_bins.append(rate)
        bad_rate_bins.append(bad_rate)
        total_bad_rate_bins.append(total_bad_rate)
        total_good_rate_bins.append(total_good_rate)
        ks_bins.append(ks)
        pass_bad_rate_bins.append(pass_bad_rate)
        refuse_bad_rate_bins.append(refuse_bad_rate)
        pass_rate_bins.append(pass_rate)

    res = {"percentile_bins": percentile_bins,
           "rate_bins": rate_bins,
           "bad_rate_bins": bad_rate_bins,
           "total_bad_rate_bins": total_bad_rate_bins,
           "total_good_rate_bins": total_good_rate_bins,
           "ks_bins": ks_bins,
           "pass_bad_rate_bins": pass_bad_rate_bins,
           "refuse_bad_rate_bins": refuse_bad_rate_bins,
           "pass_rate_bins": pass_rate_bins}

    return res
