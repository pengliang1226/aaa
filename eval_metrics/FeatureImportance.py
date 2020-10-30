# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/10 10:45
@file: FeatureImportance.py
@desc: 获取模型特征重要性
"""
from typing import Union, List, Set, Any, Dict

import numpy as np
from lightgbm import LGBMClassifier
from numpy import ndarray
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class FeatureImportance:
    def __init__(self, feature_name: Union[List, Set, ndarray] = None, estimator: Any = None,
                 importance_type: str = 'gain'):
        """
        初始化函数
        :param feature_name: 变量名称
        :param estimator: 模型
        :param importance_type: 集成模型特征重要性计算方式：weight 次数，gain 增益
        """
        assert feature_name is not None, '入模变量名称为空'
        assert estimator is not None, '模型为空'
        self.feature_name = feature_name
        self.estimator = estimator
        self.importance_type = importance_type

    def _lr_importance(self):
        feature_importance = np.around(self.estimator.coef_[0], 8)
        res = dict(zip(self.feature_name, feature_importance.tolist()))
        return res

    def _xgb_importance(self):
        # weight: 在所有树中，某特征被用来分裂节点的次数
        # gain: total_gain / weight
        # total_gain: 表示在所有树中，某特征在每次分裂节点时带来的总增益
        # cover: total_cover / weight
        # total_cover: 表示在所有树中，某特征在每次分裂节点时处理(覆盖)的所有样例的数量
        if self.importance_type == 'weight':
            res = self.estimator.get_booster().get_score(importance_type='weight')
        else:
            feature_importance = np.around(self.estimator.feature_importances_, 6)
            res = dict(zip(self.feature_name, feature_importance.tolist()))

        return res

    def _lgb_importance(self):
        weight = self.estimator.booster_.feature_importance('split')
        if self.importance_type == 'weight':
            feature_importance = weight
        else:
            feature_importance = self.estimator.booster_.feature_importance('gain') / weight
            feature_importance /= np.sum(feature_importance)
            feature_importance = np.around(feature_importance, 6)

        res = dict(zip(self.feature_name, feature_importance.tolist()))
        return res

    def _rf_importance(self):
        all_importances = np.array([tree.feature_importances_ for tree in self.estimator.estimators_ if
                                    tree.tree_.node_count > 1])
        if self.importance_type == 'weight':
            feature_importance = (all_importances > 0).sum(axis=0)
        else:
            feature_importance = np.around(self.estimator.feature_importances_, 6)

        res = dict(zip(self.feature_name, feature_importance.tolist()))
        return res

    def _gbdt_importance(self):
        trees = [tree for stage in self.estimator.estimators_ for tree in stage if tree.tree_.node_count > 1]
        all_importances = [tree.tree_.compute_feature_importances(normalize=False) for tree in trees]
        if self.importance_type == 'weight':
            feature_importance = (all_importances > 0).sum(axis=0)
        else:
            feature_importance = np.around(self.estimator.feature_importances_, 6)

        res = dict(zip(self.feature_name, feature_importance.tolist()))
        return res

    def _dt_importance(self):
        feature_importance = np.around(self.estimator.feature_importances_, 8)
        res = dict(zip(self.feature_name, feature_importance.tolist()))
        return res

    @property
    def get_result(self) -> Dict:
        if isinstance(self.estimator, LogisticRegression):
            result = self._lr_importance()
        elif isinstance(self.estimator, XGBClassifier):
            result = self._xgb_importance()
        elif isinstance(self.estimator, LGBMClassifier):
            result = self._lgb_importance()
        elif isinstance(self.estimator, DecisionTreeClassifier):
            result = self._dt_importance()
        elif isinstance(self.estimator, GradientBoostingClassifier):
            result = self._gbdt_importance()
        elif isinstance(self.estimator, RandomForestClassifier):
            result = self._rf_importance()
        else:
            raise Exception('模型为未知算法')

        return result
