# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/11/17 10:23
@file: RITrainer.py
@desc: 拒绝推断，分为数据法（Data methods）和推断法（Inference methods），数据法牵扯代码层面
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from eval_metrics import calc_ks
from model_training._base import TrainerMixin


class HardCutoff(TrainerMixin):
    """
    硬截断法（简单展开法）
    step 1. 构建KGB模型，并对全量样本打分，得到P(bad)。
    step 2. 将拒绝样本按P(bad)降序排列，设置cutoff。根据业务经验，比如拒绝样本的bad rate是放贷样本的2～4倍，从而结合拒绝样本量计算出cutoff。
    step 3. 高于cutoff的拒绝样本标记为bad，其余拒绝样本当作灰色样本，不予考虑。
    step 4. 利用组合样本构建AGB模型。
    """

    def __init__(self, estimator, IK: float = 2, flag: bool = True, file_path: str = None):
        """
        初始化函数
        :param estimator: 学习器
        :param IK: 经验风险因子，业务实际bad rate/放贷样本bad rate
        :param flag: 是否抛弃拒绝样本中的灰样本
        :param file_path: 最终建模使用样本输出路径
        """
        TrainerMixin.__init__(self)
        self.estimator = estimator
        self.IK = IK
        self.flag = flag
        self.file_path = file_path

    def fit(self, X: DataFrame, y: Series):
        """
        拟合学习器
        :param X: 包括通过样本和拒绝样本
        :param y: -1代表拒绝样本，0,1代表通过样本
        :return:
        """
        mask_kgb = (y != -1)
        mask_igb = ~mask_kgb
        X_kgb, y_kgb, X_igb = X[mask_kgb], y[mask_kgb], X[mask_igb]
        # 计算拒绝样本好坏样本数量
        bad_rate_kgb = y_kgb.sum() / len(y_kgb)
        bad_rate_igb = bad_rate_kgb * self.IK
        bad_num_igb = np.ceil(len(X_igb) * bad_rate_igb)
        good_num_igb = len(X_igb) - bad_num_igb
        # 训练KGB模型，并对拒绝样本预测
        self.estimator.fit(X_kgb, y_kgb)
        pred_igb = self.estimator.predict_proba(X_igb)[:, 1]
        # 拒绝样本打标签，并训练全量模型
        idx_sort = np.argsort(-pred_igb)
        X_igb = X_igb.iloc[idx_sort]
        if self.flag:
            X_agb = pd.concat([X_kgb, X_igb.iloc[:bad_num_igb]], ignore_index=True)
            y_agb = pd.concat([y_kgb, pd.Series(np.ones(bad_num_igb))], ignore_index=True)
        else:
            X_agb = pd.concat([X_kgb, X_igb], ignore_index=True)
            y_agb = pd.concat([y_kgb, pd.Series(np.ones(bad_num_igb)), pd.Series(np.zeros(good_num_igb))],
                              ignore_index=True)
        self.estimator.fit(X_agb, y_agb)

        if self.file_path is not None:
            pd.concat([X_agb, y_agb], axis=1).to_csv(self.file_path, index=False)


class FuzzyAugmentation(TrainerMixin):
    """
    模糊展开法
    step 1. 构建KGB模型，并对拒绝样本打分，得到P(good)和P(bad)。
    step 2. 将每条拒绝样本复制为不同类别，不同权重的两条：一条标记为good，权重为P(good)；另一条标记为bad，权重为P(bad)。
    step 3. 利用变换后的拒绝样本和放贷已知好坏样本（类别不变，权重设为1）建立AGB模型。
    """

    def __init__(self, estimator, file_path: str = None):
        """
        初始化函数
        :param estimator: 学习器, 学习器fit必须包含权重参数
        :param file_path: 最终建模使用样本输出路径
        """
        TrainerMixin.__init__(self)
        self.estimator = estimator
        self.file_path = file_path

    def fit(self, X: DataFrame, y: Series):
        """
        拟合学习器
        :param X: 包括通过样本和拒绝样本
        :param y: -1代表拒绝样本，0,1代表通过样本
        :return:
        """
        mask_kgb = (y != -1)
        mask_igb = ~mask_kgb
        X_kgb, y_kgb, X_igb = X[mask_kgb], y[mask_kgb], X[mask_igb]
        # 训练KGB模型，并对拒绝样本预测
        self.estimator.fit(X_kgb, y_kgb)
        X_igb_bad, X_igb_good = X_igb.copy(), X_igb.copy()
        X_igb_bad['weight'] = self.estimator.predict_proba(X_igb)[:, 1]
        X_igb_good['weight'] = self.estimator.predict_proba(X_igb)[:, 0]
        X_kgb['weight'] = 1

        X_agb = pd.concat([X_kgb, X_igb_good, X_igb_bad], ignore_index=True)
        y_agb = pd.concat([y_kgb, pd.Series(np.zeros(len(X_igb))), pd.Series(np.ones(len(X_igb)))], ignore_index=True)

        self.estimator.fit(X_agb.iloc[:, :-1], y_agb, sample_weight=X_agb['weight'])

        if self.file_path is not None:
            pd.concat([X_agb, y_agb], axis=1).to_csv(self.file_path, index=False)


class ReWeighting(TrainerMixin):
    """
    重新加权法
    step 1. 构建KGB模型，并对全量样本打分，得到P(bad)。
    step 2. 将全量样本按P(bad)升序排列，分箱统计每箱中的放贷和拒绝样本数。
    step 3. 计算每个分箱中放贷好坏样本的权重，weight = (reject + accept) / accept。
    step 4. 引入样本权重，利用放贷好坏样本构建KGB模型。
    """

    def __init__(self, estimator, file_path: str = None, k: int = 10):
        """
        初始化函数
        :param estimator: 学习器
        :param file_path: 最终建模使用样本输出路径
        :param k: 分箱数量
        """
        TrainerMixin.__init__(self)
        self.estimator = estimator
        self.file_path = file_path
        self.k = k

    def fit(self, X: DataFrame, y: Series):
        """
        拟合学习器
        :param X: 包括通过样本和拒绝样本
        :param y: -1代表拒绝样本，0,1代表通过样本
        :return:
        """
        mask_kgb = (y != -1)
        X_kgb, y_kgb = X[mask_kgb], y[mask_kgb]
        # 训练KGB模型，并对全部样本预测
        self.estimator.fit(X_kgb, y_kgb)
        pred_cut = pd.qcut(self.estimator.predict_proba(X)[:, 1], self.k, labels=False)

        for i in range(self.k):
            mask = (pred_cut == i)
            accept = (y[mask] != -1).sum()
            X.loc[mask, 'weight'] = mask.sum() / accept

        self.estimator.fit(X[mask_kgb].iloc[:, :-1], y_kgb, sample_weight=X['weight'])

        if self.file_path is not None:
            pd.concat([X, y], axis=1).to_csv(self.file_path, index=False)


class Extrapolation(TrainerMixin):
    """
    外推法，又称打包法
    step 1. 构建KGB模型，并对全量样本打分P(good)，也就是图中的score。
    step 2. 将放贷样本按分数排序后分箱（一般等频），将拒绝样本按相同边界分组。
    step 3. 对每个分箱，统计放贷样本中的bad rate。
    step 4. 对每个分箱，将放贷样本的bad rate乘以经验风险因子（按区间递增0.2，通常取2～4），得到拒绝样本的期望bad rate。
    step 5. 为达到期望的bad rate，利用硬截断法赋予拒绝样本以bad和good状态。同时，检验整体拒绝样本的bad rate是否是放贷样本的2～4倍。
    step 6. 利用组合样本构建AGB模型。
    """

    def __init__(self, estimator, k: int = 10, IK_down: float = 2, IK_up: float = 4, flag: bool = True,
                 file_path: str = None):
        """
        初始化函数
        :param estimator: 学习器
        :param k: 分箱数量
        :param IK_down: 经验风险因子下限，业务实际bad rate/放贷样本bad rate
        :param IK_up: 经验风险因子上限，业务实际bad rate/放贷样本bad rate
        :param flag: 是否抛弃拒绝样本中的灰样本
        :param file_path: 最终建模使用样本输出路径
        """
        TrainerMixin.__init__(self)
        self.estimator = estimator
        self.file_path = file_path
        self.k = k
        self.flag = flag
        self.interval = [IK_down + (IK_up - IK_down) / k * i for i in range(k)]

    def fit(self, X: DataFrame, y: Series):
        """
        拟合学习器
        :param X: 包括通过样本和拒绝样本
        :param y: -1代表拒绝样本，0,1代表通过样本
        :return:
        """
        mask_kgb = (y != -1)
        mask_igb = ~mask_kgb
        X_kgb, y_kgb, X_igb = X[mask_kgb], y[mask_kgb], X[mask_igb]
        # 训练KGB模型，并对全量样本预测
        self.estimator.fit(X_kgb, y_kgb)
        pred_kgb = self.estimator.predict_proba(X_kgb)[:, 1]
        pred_igb = self.estimator.predict_proba(X_igb)[:, 1]

        # 通过样本分组，并对拒绝样本使用相同分组
        _, bins = pd.qcut(pred_kgb, self.k, duplicates='drop', retbins=True)
        kgb_cut = pd.cut(pred_kgb, bins=bins)
        igb_cut = pd.cut(pred_igb, bins=bins)
        kgb_bad_rate = y_kgb.groupby(kgb_cut).apply(lambda x: x.sum() / len(x))
        igb_bad_rate = kgb_bad_rate * self.interval

        # 对拒绝样本进行打标签
        X_igb['cut'] = igb_cut
        X_igb['pred'] = pred_igb
        tmp = pd.DataFrame()
        for name, group in X_igb.groupby('cut'):
            bad_rate = igb_bad_rate[name]
            bad_num = np.ceil(group.shape[0] * bad_rate)
            group = group.sort_values(by='pred', ascending=False, ignore_index=True)
            group['y'] = 1
            if self.flag:
                group = group.iloc[:bad_num]
            else:
                group.iloc[bad_num:, -1] = 0
            tmp = tmp.append(group)

        X_igb = tmp.iloc[:, :-3]
        y_igb = tmp.loc[:, 'y']
        X_agb = pd.concat([X_kgb, X_igb], ignore_index=True)
        y_agb = pd.concat([y_kgb, y_igb], ignore_index=True)
        self.estimator.fit(X_agb, y_agb)

        if self.file_path is not None:
            pd.concat([X_agb, y_agb], axis=1).to_csv(self.file_path, index=False)


class IterativeReclassification(TrainerMixin):
    """
    迭代再分类法
    该方法通过多次迭代好坏分类，直到收敛某一临界值。拿硬截断法迭代操作步骤如下：
    step 1. 构建KGB模型，对拒绝样本打分，得到P(bad)
    step 2. 将拒绝样本按P(bad)降序排列，设置cutoff，若高于cutoff则标记为bad，反之标记为good。
    step 3. 加入推断的好坏样本，构建AGB模型，对拒绝样本打分，得到新的P(bad)。
    step 4. 迭代训练，直到模型参数收敛。如KS不再变化。
    """

    def __init__(self, estimator, method_dict: dict = None, file_path: str = None):
        """
        初始化函数
        :param estimator: 学习器
        :param method_dict: 需要进行迭代的方法参数字典，只包括硬截断法HC(method_dict={'name':,'IK':}),
                            模糊展开法FA(method_dict={'name':})，外推法Ep(method_dict={'name':,'k':,'IK_down':,'IK_up':})
        :param file_path: 最终建模使用样本输出路径
        """
        TrainerMixin.__init__(self)
        self.estimator = estimator
        assert method_dict['name'] in ['HC', 'FA', 'Ep'], 'method_dict中name参数赋值错误'
        self.method_dict = method_dict
        self.file_path = file_path

    def hard_cutoff(self, X_kgb, y_kgb, X_igb):
        """
        硬截断法迭代
        :param X_kgb: 通过样本数据
        :param y_kgb: 通过样本标签
        :param X_igb: 拒绝样本数据
        :return:
        """
        IK, flag = self.method_dict['IK'], self.method_dict['flag']

        # 计算拒绝样本好坏样本数量
        bad_rate_kgb = y_kgb.sum() / len(y_kgb)
        bad_rate_igb = bad_rate_kgb * IK
        bad_num_igb = np.ceil(len(X_igb) * bad_rate_igb)
        good_num_igb = len(X_igb) - bad_num_igb

        KS_max = 0
        n = 0
        X_agb, y_agb = pd.DataFrame(), pd.DataFrame()
        while True:
            pred_kgb = self.estimator.predict_proba(X_kgb)[:, 1]
            ks, _ = calc_ks(y_kgb, pred_kgb)
            if KS_max > ks:
                break
            else:
                print('第{}轮ks值为{}'.format(n, ks))
                KS_max = ks
                n += 1

            pred_igb = self.estimator.predict_proba(X_igb)[:, 1]
            idx_sort = np.argsort(-pred_igb)
            if flag:
                X_agb = pd.concat([X_kgb, X_igb.iloc[idx_sort[:bad_num_igb]]], ignore_index=True)
                y_agb = pd.concat([y_kgb, pd.Series(np.ones(bad_num_igb))], ignore_index=True)
            else:
                X_agb = pd.concat([X_kgb, X_igb.iloc[idx_sort]], ignore_index=True)
                y_agb = pd.concat([y_kgb, pd.Series(np.ones(bad_num_igb)), pd.Series(np.zeros(good_num_igb))],
                                  ignore_index=True)
            self.estimator.fit(X_agb, y_agb)

        if self.file_path is not None:
            pd.concat([X_agb, y_agb], axis=1).to_csv(self.file_path, index=False)

    def fuzzy_augmentation(self, X_kgb, y_kgb, X_igb):
        """
        模糊展开法迭代
        :param X_kgb: 通过样本数据
        :param y_kgb: 通过样本标签
        :param X_igb: 拒绝样本数据
        :return:
        """
        KS_max = 0
        n = 0
        X_kgb['weight'] = 1
        X_agb, y_agb = pd.DataFrame(), pd.DataFrame()
        while True:
            pred_kgb = self.estimator.predict_proba(X_kgb.iloc[:, :-1])[:, 1]
            ks, _ = calc_ks(y_kgb, pred_kgb)
            if KS_max > ks:
                break
            else:
                print('第{}轮ks值为{}'.format(n, ks))
                KS_max = ks
                n += 1

            X_igb_bad, X_igb_good = X_igb.copy(), X_igb.copy()
            X_igb_bad['weight'] = self.estimator.predict_proba(X_igb)[:, 1]
            X_igb_good['weight'] = self.estimator.predict_proba(X_igb)[:, 0]

            X_agb = pd.concat([X_kgb, X_igb_good, X_igb_bad], ignore_index=True)
            y_agb = pd.concat([y_kgb, pd.Series(np.zeros(len(X_igb))), pd.Series(np.ones(len(X_igb)))],
                              ignore_index=True)

            self.estimator.fit(X_agb.iloc[:, :-1], y_agb, sample_weight=X_agb['weight'])

        if self.file_path is not None:
            pd.concat([X_agb, y_agb], axis=1).to_csv(self.file_path, index=False)

    def extrapolation(self, X_kgb, y_kgb, X_igb):
        """
        外推法迭代
        :param X_kgb: 通过样本数据
        :param y_kgb: 通过样本标签
        :param X_igb: 拒绝样本数据
        :return:
        """
        IK_down, IK_up, k = self.method_dict['IK_down'], self.method_dict['IK_up'], self.method_dict['k']
        flag = self.method_dict['flag']
        interval = [IK_down + (IK_up - IK_down) / k * i for i in range(k)]

        KS_max = 0
        n = 0
        X_agb, y_agb = pd.DataFrame(), pd.DataFrame()
        while True:
            pred_kgb = self.estimator.predict_proba(X_kgb)[:, 1]
            ks, _ = calc_ks(y_kgb, pred_kgb)
            if KS_max > ks:
                break
            else:
                print('第{}轮ks值为{}'.format(n, ks))
                KS_max = ks
                n += 1

            pred_igb = self.estimator.predict_proba(X_igb)[:, 1]
            # 通过样本分组，并对拒绝样本使用相同分组
            _, bins = pd.qcut(pred_kgb, k, duplicates='drop', retbins=True)
            kgb_cut = pd.cut(pred_kgb, bins=bins)
            igb_cut = pd.cut(pred_igb, bins=bins)
            kgb_bad_rate = y_kgb.groupby(kgb_cut).apply(lambda x: x.sum() / len(x))
            igb_bad_rate = kgb_bad_rate * interval

            # 对拒绝样本进行打标签
            X_igb_copy = X_igb.copy()
            X_igb_copy['cut'] = igb_cut
            X_igb_copy['pred'] = pred_igb
            tmp = pd.DataFrame()
            for name, group in X_igb_copy.groupby('cut'):
                bad_rate = igb_bad_rate[name]
                bad_num = np.ceil(group.shape[0] * bad_rate)
                group = group.sort_values(by='pred', ascending=False, ignore_index=True)
                group['y'] = 1
                if flag:
                    group = group.iloc[:bad_num]
                else:
                    group.iloc[bad_num:, -1] = 0
                tmp = tmp.append(group)

            X_agb = pd.concat([X_kgb, tmp.iloc[:, :-3]], ignore_index=True)
            y_agb = pd.concat([y_kgb, tmp.loc[:, 'y']], ignore_index=True)
            self.estimator.fit(X_agb, y_agb)

        if self.file_path is not None:
            pd.concat([X_agb, y_agb], axis=1).to_csv(self.file_path, index=False)

    def fit(self, X: DataFrame, y: Series):
        """
        拟合学习器
        :param X: 包括通过样本和拒绝样本
        :param y: -1代表拒绝样本，0,1代表通过样本
        :return:
        """
        mask_kgb = (y != -1)
        mask_igb = ~mask_kgb
        X_kgb, y_kgb, X_igb = X[mask_kgb], y[mask_kgb], X[mask_igb]
        self.estimator.fit(X_kgb, y_kgb)

        method_name = self.method_dict['name']
        if method_name == 'HC':
            self.hard_cutoff(X_kgb, y_kgb, X_igb)
        elif method_name == 'FA':
            self.fuzzy_augmentation(X_kgb, y_kgb, X_igb)
        elif method_name == 'Ep':
            self.extrapolation(X_kgb, y_kgb, X_igb)
