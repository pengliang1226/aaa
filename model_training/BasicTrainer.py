# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/17 10:21
@file: BasicTrainer.py
@desc: 模型训练，算法名称缩写lr, gbdt, rf, lgb, xgb
"""
from typing import Dict

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from model_training._base import TrainerMixin


class BasicTrainer(TrainerMixin):
    def __init__(self, algorithm: str = None, params: Dict = None):
        """
        初始化函数
        :param algorithm: 算法名称, 缩写lr, gbdt, rf, lgb, xgb
        :param params: 算法参数
        """
        TrainerMixin.__init__(self)
        self.algorithm = algorithm
        self.params = params

    def create_estimator(self):
        """
        创建学习器
        :return:
        """
        if self.algorithm == 'lr':
            self.estimator = LogisticRegression()
        elif self.algorithm == 'gbdt':
            self.estimator = GradientBoostingClassifier()
        elif self.algorithm == 'rf':
            self.estimator = RandomForestClassifier()
        elif self.algorithm == 'lgb':
            self.estimator = LGBMClassifier()
        else:
            self.estimator = XGBClassifier()

        if self.params is not None:
            for k, v in self.params.items():
                if not hasattr(self.estimator, k):
                    raise Exception('算法{}不包含参数{}'.format(self.algorithm, k))
                else:
                    setattr(self.estimator, k, v)
