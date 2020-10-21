import math
import os
from pickle import load, dump
from typing import Any, Union, Dict, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

from util.config import DISCRETE, CONTINUOUS

__SMOOTH__ = 1e-6
__DEFAULT__ = 1e-6


def parse_float(s: Any) -> Union[str, float]:
    if s is None:
        return None
    try:
        res = float(s)
    except (TypeError, ValueError):
        res = s
    return res


def load_pkl(pkl_file: str = None):
    """
    Load object from pickle file.

    :param pkl_file: pickle file
    :return: original python object
    """
    if pkl_file is None:
        raise Exception('模型文件地址为空')
    if not os.path.exists(pkl_file):
        raise Exception("模型文件不存在")
    with open(pkl_file, 'rb') as f:
        res = load(f)
    return res


def dump_pkl(obj: Any, pkl_file: str = None):
    """
    Pickling the python object to file.

    :param obj: original python object
    :param pkl_file: output models file
    :return: None (IO side effect: generate a pkl file)
    """
    if pkl_file is None:
        raise Exception('模型文件地址为空')
    with open(pkl_file, 'wb') as f:
        dump(obj, f)


def woe_single(B: float, G: float, b: float, g: float) -> float:
    """
    woe计算
    """
    res = np.log(((b + __SMOOTH__) / (B + __SMOOTH__)) / ((g + __SMOOTH__) / (G + __SMOOTH__)))
    return __DEFAULT__ if math.isinf(res) else res


def woe_single_all(B: float, G: float, b: ndarray, g: ndarray) -> ndarray:
    """
    woe计算
    """
    res = np.log(((b + __SMOOTH__) / (B + __SMOOTH__)) / ((g + __SMOOTH__) / (G + __SMOOTH__)))
    res = np.where(np.isinf(res), __DEFAULT__, res)
    return res


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


def divide_train_test(data: DataFrame, flag_y: Any, seed: int = None, test_size: float = 0.3):
    """
    样本划分
    :param data: 数据
    :param flag_y: y标签列名
    :param seed: 随机种子
    :param test_size: 切分比例
    :return:
    """
    x = data.loc[:, data.columns != flag_y]
    y = data.loc[:, data.columns == flag_y]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    x_train.insert(0, flag_y, y_train)
    x_test.insert(0, flag_y, y_test)
    return x_train, x_test


def woe_transform(data: DataFrame, woe_dict: Dict):
    """
    样本数据转woe
    :param data: 数据
    :param woe_dict: woe转码信息
    :return:
    """
    for key, value in woe_dict.items():
        if value == {}:
            continue
        col = data[key]
        bins = value['bins'].split(';')
        woe_bins = value['woe_bins'].split(',')
        for i, b in enumerate(bins):
            if '[' in b and ']' in b:
                b = eval(b)
                col[col.isin(b)] = float(woe_bins[i])
            elif '(' in b:
                left = b[1:-1].split(',')[0]
                right = b[1:-1].split(',')[1]
                if left == '-inf':
                    col[col <= float(right)] = float(woe_bins[i])
                elif right == 'inf':
                    col[col > float(left)] = float(woe_bins[i])
                else:
                    col[(col > float(left)) & (col <= float(right))] = float(woe_bins[i])
            else:
                left = b[1:-1].split(',')[0]
                right = b[1:-1].split(',')[1]
                if left == '-inf':
                    col[col < float(right)] = float(woe_bins[i])
                elif right == 'inf':
                    col[col >= float(left)] = float(woe_bins[i])
                else:
                    col[(col >= float(left)) & (col < float(right))] = float(woe_bins[i])


def get_feature_type(X: Series, threshold: int = 10, null_value: List = None) -> Dict:
    """
    获取变量属性类型；定性或定量
    :param X: 变量数据
    :param threshold: 变量唯一值个数阈值，用来针对int型变量小于阈值则定义为定性变量（离散变量）
    :param null_value: 缺失值标识符, list可能存在多个缺失值
    :return:
    """
    if null_value is not None:
        X.replace(null_value, [np.nan] * len(null_value), inplace=True)
    data_type = X.convert_dtypes().dtype.name
    unique_num = X[X.notna()].unique().size
    if (data_type == 'Int64' and unique_num >= threshold) or data_type == 'float64':
        return CONTINUOUS
    else:
        return DISCRETE


def one_hot_encoding(X: Series, feature_name: Any, null_value: Any = None) -> DataFrame:
    """
    定性变量one-hot转码
    :param X: 变量数据
    :param feature_name: 变量名称，用来对转码后的变量命名
    :param null_value: 缺失值标识符
    :return: 返回转码后的DataFrame
    """
    unique = X[X.notna()].unique()
    if null_value is not None:
        unique = np.delete(unique, np.where(unique == null_value))
    data_numpy = np.where(unique == X[:, None], 1, 0)
    data_cols = [feature_name + '_' + str(x) for x in unique]
    data = pd.DataFrame(data_numpy, columns=data_cols)
    return data


def woe_encoding(X: Series, y: Series, null_value: Any = None, bad_y: Any = 1) -> Series:
    """
    定性变量woe有序转码，根据woe从小到大排序，替换为0-n数字
    :param X: 变量数据
    :param y: y标签数据
    :param null_value: 缺失值标识符
    :param bad_y: 坏样本值
    :return: 返回转码后的Series
    """
    if null_value is not None:
        X.fillna(null_value, inplace=True)

    B = (y == bad_y).sum()
    G = y.size - B
    unique_value = X.unique()
    mask = (unique_value.reshape(-1, 1) == X.values)
    mask_bad = mask & (y.values == bad_y)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    woe_value = np.around(woe_single_all(B, G, b, g), 6)
    woe_value_sort = np.argsort(woe_value)
    X_encode = X.map(dict(zip(unique_value, woe_value_sort)))
    return X_encode


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


def iv_part(x: Union[Series, ndarray], y: Union[Series, ndarray]) -> Dict:
    """
    计算单个变量中每个分箱的iv值
    :param x: 单个变量数据
    :param y: 标签数据
    :return:
    """
    if isinstance(x, Series):
        x = x.values
    if isinstance(y, Series):
        y = y.values
    B = y.sum()
    G = y.size - B
    unique_value = np.unique(x)
    mask = (unique_value.reshape(-1, 1) == x)
    mask_bad = mask & (y == 1)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    temp = (b + __SMOOTH__) / (B + __SMOOTH__) - (g + __SMOOTH__) / (G + __SMOOTH__)
    iv_value = np.around(temp * woe_single_all(B, G, b, g), 6)
    iv_value = iv_value.tolist()
    res = dict(zip(unique_value, iv_value))
    return res


def woe_part(x: Union[Series, ndarray], y: Union[Series, ndarray]) -> Dict:
    """
    计算单个变量每个分箱的woe
    :param x: 单个变量数据
    :param y: 标签数据
    :return:
    """
    if isinstance(x, Series):
        x = x.values
    if isinstance(y, Series):
        y = y.values
    B = y.sum()
    G = y.size - B
    unique_value = np.unique(x)
    mask = (unique_value.reshape(-1, 1) == x)
    mask_bad = mask & (y == 1)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    woe_value = np.around(woe_single_all(B, G, b, g), 6)
    woe_value = woe_value.tolist()
    res = dict(zip(unique_value, woe_value))
    return res
