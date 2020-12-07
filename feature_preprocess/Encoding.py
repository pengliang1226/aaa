# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/12/7 11:09
@file: Encoding.py
@desc: 特征编码
"""
from typing import List

from category_encoders import OrdinalEncoder, OneHotEncoder, HashingEncoder, HelmertEncoder, SumEncoder, \
    TargetEncoder, MEstimateEncoder, JamesSteinEncoder, WOEEncoder, LeaveOneOutEncoder, CatBoostEncoder
from pandas import DataFrame, Series
from sklearn.base import TransformerMixin


class FeatureEncoding(TransformerMixin):
    def __init__(self, cols: List = None):
        """
        初始化函数
        :param cols: 编码列列表
        """
        self.cols = cols
        self.encoder = None

    def Ordinal_Encoding(self):
        """
        序数编码将类别变量转化为一列序数变量，包含从1到类别数量之间的整数
        :return:
        """
        self.encoder = OrdinalEncoder(cols=self.cols)

    def OneHot_Encoding(self, handle_missing='indicator', handle_unknown='indicator'):
        """
        one-hot编码，其可以将具有n_categories个可能值的一个分类特征转换为n_categories个二进制特征，其中一个为1，所有其他为0
        :param handle_missing: 默认value，缺失值用全0替代；indicator，增加缺失值一列
        :param handle_unknown: 默认value，未知值用全0替代；indicator，增加未知值一列
        :return:
        """
        self.encoder = OneHotEncoder(cols=self.cols, handle_missing=handle_missing, handle_unknown=handle_unknown)

    def Hashing_Encoding(self, n_components: int = 8):
        """
        哈希编码，将任意数量的变量以一定的规则映射到给定数量的变量。特征哈希可能会导致要素之间发生冲突。哈希编码器的大小及复杂程度不随数据类别的增多而增多。
        :param n_components: 用来表示特征的位数
        :return:
        """
        self.encoder = HashingEncoder(cols=self.cols, n_components=n_components)

    def Helmert_Encoding(self, handle_missing='indicator', handle_unknown='indicator'):
        """
        Helmert编码，分类特征中的每个值对应于Helmert矩阵中的一行
        :param handle_missing: 默认value，缺失值用全0替代；indicator，增加缺失值一列
        :param handle_unknown: 默认value，未知值用全0替代；indicator，增加未知值一列
        :return:
        """
        self.encoder = HelmertEncoder(cols=self.cols, handle_unknown=handle_unknown, handle_missing=handle_missing)

    def Devaition_Encoding(self, handle_missing='indicator', handle_unknown='indicator'):
        """
        偏差编码。偏差编码后，线性模型的系数可以反映该给定该类别变量值的情况下因变量的平均值与全局因变量的平均值的差异
        :param handle_missing: 默认value，缺失值用全0替代；indicator，增加缺失值一列
        :param handle_unknown: 默认value，未知值用全0替代；indicator，增加未知值一列
        :return:
        """
        self.encoder = SumEncoder(cols=self.cols, handle_missing=handle_missing, handle_unknown=handle_unknown)

    def Target_Encoding(self, min_samples_leaf: int = 1, smoothing: float = 1.0):
        """
        目标编码是一种不仅基于特征值本身，还基于相应因变量的类别变量编码方法。
        对于分类问题：将类别特征替换为给定某一特定类别值的因变量后验概率与所有训练数据上因变量的先验概率的组合。
        对于连续目标：将类别特征替换为给定某一特定类别值的因变量目标期望值与所有训练数据上因变量的目标期望值的组合。
        该方法严重依赖于因变量的分布，但这大大减少了生成编码后特征的数量。
        :param min_samples_leaf:
        :param smoothing:
        :return:
        """
        self.encoder = TargetEncoder(cols=self.cols, min_samples_leaf=min_samples_leaf, smoothing=smoothing)

    def MEstimate_Encoding(self, m: float = 1.0, sigma: float = 0.05, randomized: bool = False):
        """
        M估计量编码是目标编码的一个简化版本
        :param m:
        :param sigma:
        :param randomized:
        :return:
        """
        self.encoder = MEstimateEncoder(cols=self.cols, m=m, sigma=sigma, randomized=randomized)

    def JamesStein_Encoding(self, model: str = 'independent', sigma: float = 0.05, randomized: bool = False):
        """
        James-Stein编码，也是一种基于目标编码的编码方法，也尝试通过参数B来平衡先验概率与观测到的条件概率。
        但与目标编码与M估计量编码不同的是，James-Stein编码器通过方差比而不是样本大小来平衡两个概率。
        :param model:
        :param sigma:
        :param randomized:
        :return:
        """
        self.encoder = JamesSteinEncoder(cols=self.cols, model=model, sigma=sigma, randomized=randomized)

    def WOE_Encoding(self, regularization: float = 1.0, sigma: float = 0.05, randomized: bool = False):
        """
        woe编码
        :param regularization:
        :param sigma:
        :param randomized:
        :return:
        """
        self.encoder = WOEEncoder(cols=self.cols, regularization=regularization, randomized=randomized, sigma=sigma)

    def LeaveOneOut_Encoding(self, sigma: float = 0.05):
        """
        留一编码
        :param sigma:
        :return:
        """
        self.encoder = LeaveOneOutEncoder(cols=self.cols, sigma=sigma)

    def CatBoost_Encoding(self, sigma: float = None, a: float = 1):
        """
        CatBoost是一个基于树的梯度提升模型。其在包含大量类别特征的数据集问题中具有出色的效果。
        在使用Catboost编码器之前，必须先对训练数据随机排列，因为在Catboost中，编码是基于“时间”的概念，即数据集中观测值的顺序。
        :param sigma:
        :param a:
        :return:
        """
        self.encoder = CatBoostEncoder(cols=self.cols, a=a, sigma=sigma)

    def fit(self, X: DataFrame, y: Series = None):
        """
        拟合函数
        :param X:
        :param y:
        :return:
        """
        if y is None:
            self.encoder.fit(X)
        else:
            self.encoder.fit(X, y)

    def transform(self, X: DataFrame):
        """
        转换函数
        :param X:
        :return:
        """
        res = self.encoder.transform(X)

        return res
