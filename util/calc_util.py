import os
from math import isinf, inf
from pickle import load, dump
from typing import Any, Union, Dict, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    return __DEFAULT__ if isinf(res) else res


def woe_single_all(B: float, G: float, b: ndarray, g: ndarray) -> ndarray:
    """
    woe计算
    """
    res = np.log(((b + __SMOOTH__) / (B + __SMOOTH__)) / ((g + __SMOOTH__) / (G + __SMOOTH__)))
    res = np.where(np.isinf(res), __DEFAULT__, res)
    return res


def divide_sample(df: DataFrame, seed: int = None, test_size: float = 0.2):
    """
    样本划分
    :param df: 数据
    :param seed: 随机种子
    :param test_size: 切分比例
    :return:
    """
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=seed)
    return train_data, test_data


def woe_transform(X: Series, feat_type: int, bins_info: Dict, woes_info: List):
    """
    样本数据转woe
    :param X: 变量数据
    :param feat_type: 变量属性类型
    :param bins_info: 分箱信息
    :param woes_info: woe信息
    :return:
    """
    bins_mask = []
    bins = bins_info['bins']
    flag = bins_info['flag']
    X_trans = X.copy()
    if flag == 1:
        bins_mask.append(X_trans.isin(bins[0]))
        X_trans.loc[X_trans.isin(bins[0])] = np.nan
        bins = bins[1:]

    if feat_type == 1:
        for left, right in bins:
            mask = (X_trans > left) & (X_trans <= right)
            bins_mask.append(mask)
    else:
        for v in bins:
            mask = X_trans.isin(v)
            bins_mask.append(mask)

    mask_untrans = bins_mask[0]
    for i, mask in enumerate(bins_mask):
        mask_untrans = (mask_untrans | mask)
        X_trans.loc[mask] = float(woes_info[i])
    # 对分箱区间不包含的值，替换为首个区间的woe
    if not all(mask_untrans):
        X_trans.loc[~mask_untrans] = float(woes_info[0]) if flag == 0 else float(woes_info[1])

    return X_trans.astype('float64')


def get_attr_by_unique(X: Series, threshold: int = 10, null_value: List = None) -> Dict:
    """
    通过唯一值个数确定变量属性类型；定性或定量
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


def get_attr_by_dtype(X: Series) -> Dict:
    """
    通过列属性确定变量属性类型；定性或定量
    :param X: 变量数据
    :return:
    """
    data_type = X.convert_dtypes().dtype.name
    if data_type == 'Int64' or data_type == 'float64':
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


def disorder_mapping(col_data: Series, y: Series, bad_y: Any = 1, null_value: List = None) -> Dict:
    """
    无序变量转码
    :param col_data: 变量数据
    :param y: y标签数据
    :param bad_y: 坏样本值
    :param null_value: 缺失值
    :return:
    """
    mask = col_data.isin(null_value)
    x = col_data[~mask]
    y = y[~mask]
    B = (y == bad_y).sum()
    G = y.size - B
    unique_value = x.unique()
    mask = (unique_value.reshape(-1, 1) == x.values)
    mask_bad = mask & (y.values == bad_y)
    b = mask_bad.sum(axis=1)
    g = mask.sum(axis=1) - b
    woe_value = np.around(woe_single_all(B, G, b, g), 6)
    woe_value_sort = np.argsort(woe_value)
    x = x.map(dict(zip(unique_value, woe_value_sort)))
    tree = DecisionTreeClassifier(max_leaf_nodes=6, min_samples_leaf=max(int(x.size * 0.05), 50))
    tree.fit(x.values.reshape(-1, 1), y)
    threshold = [-inf]
    threshold.extend(np.sort(tree.tree_.threshold[tree.tree_.feature == 0]).tolist())
    threshold.append(inf)
    index = pd.cut(woe_value_sort, threshold, right=True, include_lowest=True, labels=False)
    res = dict(zip(unique_value.tolist(), index.tolist()))
    for k in null_value:
        res[k] = k

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
