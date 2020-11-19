# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/17 10:21
@file: BasicTrainer.py
@desc: 模型训练，算法名称缩写lr, gbdt, rf, lgb, xgb
"""
from typing import Dict

import numpy as np
from lightgbm import LGBMClassifier
from pandas import DataFrame, Series
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from eval_metrics.BinaryClassifier import calc_ks


class BasicTrainer:
    def __init__(self, algorithm: str = None, params: Dict = None):
        """
        初始化函数
        :param algorithm: 算法名称, 缩写lr, gbdt, rf, lgb, xgb
        :param params: 算法参数
        """
        self.algorithm = algorithm
        self.params = params
        self.estimator = None

    def get_estimator(self):
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

    def fit(self, X: DataFrame, y: Series):
        """
        拟合学习器
        :param X:
        :param y:
        :return:
        """
        self.get_estimator()
        self.estimator.fit(X, y)

    def fit_Kfold(self, X: DataFrame, y: Series, k: int = 5):
        """
        交叉验证拟合最优学习器
        :param X:
        :param y:
        :param k: 交叉验证折数
        :return:
        """
        metrics, models = [], []
        self.get_estimator()
        estimator = self.estimator
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        kf_list = list(kf.split(X))
        for i, index in enumerate(kf_list):
            # 训练
            train_index, test_index = index[0], index[1]
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            if self.algorithm == 'xgb':
                estimator.fit(X_train, y_train,
                              early_stopping_rounds=50,
                              eval_set=[(X_test, y_test)],
                              eval_metric='auc')
            elif self.algorithm == 'lgb':
                estimator.fit(X_train, y_train,
                              early_stopping_rounds=50,
                              eval_set=[(X_test, y_test)],
                              eval_metric='auc')
            else:
                estimator.fit(X_train, y_train)
            pred_train = estimator.predict_proba(X_train)[:, 1]
            pred_test = estimator.predict_proba(X_test)[:, 1]
            # 计算指标
            ks_train, ks_test = calc_ks(y_train, pred_train)[0], calc_ks(y_test, pred_test)[0]
            tmp = 1.8 * ks_test - 0.8 * abs(ks_train - ks_test)
            metrics.append(tmp)
            models.append(estimator)

        best_model_idx = np.argmax(metrics)
        self.estimator = models[best_model_idx]