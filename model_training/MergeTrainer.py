# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/25 10:20
@file: MergeTrainer.py
@desc: 模型融合训练，原理可以参考https://zhuanlan.zhihu.com/p/61705517
"""
from typing import List

from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier, StackingCVClassifier

from model_training._base import TrainerMixin


class VotingTrainer(TrainerMixin):
    def __init__(self, classifiers: List = None, voting: str = 'soft', weights: List = None):
        """
        投票法-初始化函数
        :param classifiers: 模型融合使用分类算法列表，内含学习器, 例：[clf1,clf2,clf3]
        :param voting: 投票法参数，投票方式。'soft': 使用类概率的投票, 'hard': 使用类标签的投票
        :param weights: 投票法参数，学习器权重. 例：[2,1,1] 代表学习器权重比值
        :return:
        """
        assert classifiers is not None, 'classifiers列表为空'
        TrainerMixin.__init__(self)
        self.classifiers = classifiers
        self.voting = voting
        self.weights = weights
        self.create_estimator()

    def create_estimator(self):
        self.estimator = EnsembleVoteClassifier(clfs=self.classifiers,
                                                voting=self.voting,
                                                weights=self.weights)


class StackingTrainer(TrainerMixin):
    def __init__(self, classifiers: List = None, meta_classifier=None, use_probas=True, average_probas=False):
        """
        stacking-初始化函数
        将训练好的所有基模型对整个训练集进行预测，第j个基模型对第i个训练样本的预测值将作为新的训练集中第i个样本的第j个特征值，
        最后基于新的训练集进行训练。同理，预测的过程也要先经过所有基模型的预测形成新的测试集，最后再对测试集进行预测
        :param classifiers: 模型融合使用分类算法列表，内含学习器, 例：[clf1,clf2,clf3]
        :param meta_classifier: 基模型预测结果，采用该分类器进行训练
        :param use_probas: True, 使用基模型预测概率列表作为结果，否则使用基学习器预测类别作为结果
        :param average_probas: True, 那么这些基分类器对每一个类别产生的概率值会被平均，否则会拼接
        :return:
        """
        assert classifiers is not None, 'classifiers参数为空'
        assert meta_classifier is not None, 'meta_classifier参数为空'
        TrainerMixin.__init__(self)
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probas = use_probas
        self.average_probas = average_probas
        self.create_estimator()

    def create_estimator(self):
        self.estimator = StackingClassifier(classifiers=self.classifiers,
                                            meta_classifier=self.meta_classifier,
                                            use_probas=self.use_probas,
                                            average_probas=self.average_probas)


class StackingCVTrainer(TrainerMixin):
    def __init__(self, classifiers: List = None, meta_classifier=None, use_probas=True, cv: int = 5):
        """
        stackingCV-初始化函数, 相对于stack增加交叉验证
        :param classifiers: 模型融合使用分类算法列表，内含学习器, 例：[clf1,clf2,clf3]
        :param meta_classifier: 基模型预测结果，采用该分类器进行训练
        :param use_probas: True, 使用基模型预测概率列表作为结果，否则使用基学习器预测类别作为结果
        :param cv: 交叉验证折数
        :return:
        """
        assert classifiers is not None, 'classifiers参数为空'
        assert meta_classifier is not None, 'meta_classifier参数为空'
        TrainerMixin.__init__(self)
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probas = use_probas
        self.cv = cv
        self.create_estimator()

    def create_estimator(self):
        self.estimator = StackingCVClassifier(classifiers=self.classifiers,
                                              meta_classifier=self.meta_classifier,
                                              use_probas=self.use_probas,
                                              cv=self.cv)
