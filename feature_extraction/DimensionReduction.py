# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/12/1 14:03
@file: DimensionReduction.py
@desc: 特征降维
"""
from typing import Union

import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, MiniBatchSparsePCA, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DimensionReduction(TransformerMixin):
    def __init__(self, n_components: Union[float, int, str] = None):
        """
        初始化函数
        [注]: 降维之前可能需要进行标准化
        :param n_components: 保留列数
        """
        self.n_components = n_components
        self.compressor = None

    def PCA(self, svd_solver: str = 'auto', **kwargs):
        """
        PCA降维
        :param svd_solver: 指定奇异值分解SVD的方法，{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。
        :param kwargs: 根据个人需求选择参数
        :return:
        """
        kwargs['n_components'] = self.n_components
        kwargs['svd_solver'] = svd_solver
        self.compressor = PCA(kwargs)

    def KernelPCA(self, kernel: str = 'linear', **kwargs):
        """
        核函数PCA降维
        [注]: 主要用于非线性数据的降维，需要用到核技巧
        :param kernel: 核函数
        :param kwargs: 根据个人需求选择参数
        :return:
        """
        assert self.n_components is None or isinstance(self.n_components, int), '参数n_components只能为None或int类型'
        kwargs['n_components'] = self.n_components
        kwargs['kernel'] = kernel
        self.compressor = KernelPCA(kwargs)

    def IncrementalPCA(self, batch_size: int = None, **kwargs):
        """
        IncrementalPCA, 主要是为了解决单机内存限制的
        :param batch_size: 每个batch样本数
        :param kwargs: 根据个人需求选择参数
        :return:
        """
        assert self.n_components is None or isinstance(self.n_components, int), '参数n_components只能为None或int类型'
        kwargs['n_components'] = self.n_components
        kwargs['batch_size'] = batch_size
        self.compressor = IncrementalPCA(kwargs)

    def SparsePCA(self, alpha: float = 1, **kwargs):
        """
        SparsePCA降维
        [注]: 主要是使用了L1的正则化，这样可以将很多非主要成分的影响度降为0，这样在PCA降维的时候我们仅仅需要对那些相对比较主要的成分进行PCA降维，
        避免了一些噪声之类的因素对我们PCA降维的影响。
        :param alpha:
        :param kwargs: 根据个人需求选择参数
        :return:
        """
        assert self.n_components is None or isinstance(self.n_components, int), '参数n_components只能为None或int类型'
        kwargs['n_components'] = self.n_components
        kwargs['alpha'] = alpha
        self.compressor = SparsePCA(kwargs)

    def MiniBatchSparsePCA(self, alpha: float = 1, batch_size: int = None, **kwargs):
        """
        MiniBatchSparsePCA降维
        [注]: 主要是使用了L1的正则化，这样可以将很多非主要成分的影响度降为0，MiniBatchSparsePCA通过使用一部分样本特征和给定的迭代次数来进行PCA降维，
        以解决在大样本时特征分解过慢的问题，代价就是PCA降维的精确度可能会降低。
        :param alpha:
        :param batch_size:
        :param kwargs: 根据个人需求选择参数
        :return:
        """
        assert self.n_components is None or isinstance(self.n_components, int), '参数n_components只能为None或int类型'
        kwargs['n_components'] = self.n_components
        kwargs['batch_size'] = batch_size
        kwargs['alpha'] = alpha
        self.compressor = MiniBatchSparsePCA(kwargs)

    def LDA(self, solver: str = 'svd', **kwargs):
        """
        线性判别分析LDA
        :param solver: 求解算法，取值可以为: svd: 使用奇异值分解求解，不用计算协方差矩阵，适用于特征数量很大的情形
                                         lsqr: 最小平方QR分解，可以结合shrinkage使用，
                                         eigen: 特征值分解，可以结合shrinkage使用
        :return:
        """
        assert self.n_components is None or isinstance(self.n_components, int), '参数n_components只能为None或int类型'
        kwargs['n_components'] = self.n_components
        kwargs['solver'] = solver
        self.compressor = LinearDiscriminantAnalysis(kwargs)

    def fit(self, X: DataFrame, y: Series = None):
        """
        拟合函数
        :param X:
        :param y:
        :return:
        """
        if y is None:
            self.compressor.fit(X)
        else:
            self.compressor.fit(X, y)

    def transform(self, X: DataFrame):
        """
        转换函数
        :param X:
        :return:
        """
        res = self.compressor.transform(X)
        res = pd.DataFrame(res, columns=X.columns)

        return res
