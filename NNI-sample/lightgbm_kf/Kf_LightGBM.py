import json
import logging

import nni
import numpy as np
import pandas as pd
from cal_metrics.cal_metrics import calc_ks, calc_auc, calc_gini, calc_psi, dump_pkl, write_data
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn2pmml import PMMLPipeline

LOG = logging.getLogger('lightgbm-classification')

with open("./search_space.json", 'rb') as f:
    search_space = json.load(f)


def load_data(file_path='./model_data.fea'):
    """
    Load data from file system.

    :param file_path:
    :return:
    """
    model_data = pd.read_feather(file_path)
    model_data.set_index('autoindex', inplace=True)
    kf = KFold(n_splits=search_space.get('KFold_num', 5), shuffle=True, random_state=4321)
    return kf, model_data


def get_default_parameters():
    """
    Get model according to given parameters.

    :return:
    """
    params = {
        # 默认参数
        "objective": 'binary',
        "n_jobs": -1,
        "boosting_type": 'gbdt',
        # 更新参数
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_split_gain": 0.0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 0
    }
    return params


def get_model(PARAMS):
    """
    Get model according to given parameters.

    :param PARAMS:
    :return:
    """
    estimator = LGBMClassifier()
    for k in PARAMS:
        if hasattr(estimator, k):
            setattr(estimator, k, PARAMS.get(k))
    pipeline = PMMLPipeline([('estimator', estimator)])
    return pipeline


def run(kf, data, model, label='flag_y'):
    """
    Train model, predict on test set and get model performance.

    :param kf:
    :param data:
    :param model:
    :param label:
    :return:
    """
    defaults, gini_trains, gini_tests, auc_trains, auc_tests, ks_trains, ks_tests, psis, models = [], [], [], [], [], \
                                                                                                  [], [], [], []
    # 交叉验证
    kf_list = list(kf.split(data))
    for i, index in enumerate(kf_list):
        # 训练
        X = data.drop(columns=label)
        y = data[label]
        train_index, test_index = index[0], index[1]
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        model.fit(X_train, y_train,
                  estimator__early_stopping_rounds=50,
                  estimator__eval_set=[(X_test, y_test)],
                  estimator__eval_metric='auc')
        y_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
        # 计算指标
        ks_train, ks_test = calc_ks(y_pred, y_train), calc_ks(y_test_pred, y_test)
        auc_train, auc_test = calc_auc(y_pred, y_train), calc_auc(y_test_pred, y_test)
        gini_train, gini_test = calc_gini(y_pred, y_train), calc_gini(y_test_pred, y_test)
        psi = calc_psi(y_pred, y_test_pred)
        default = 1.8 * ks_test - 0.8 * abs(ks_train - ks_test)
        defaults.append(default)
        gini_trains.append(gini_train)
        gini_tests.append(gini_test)
        auc_trains.append(auc_train)
        auc_tests.append(auc_test)
        ks_trains.append(ks_train)
        ks_tests.append(ks_test)
        psis.append(psi)
        models.append(model)
    # 整合结果
    metrics = {
        'gini_train': float(np.mean(gini_trains)),
        'gini_test': float(np.mean(gini_tests)),
        'auc_train': float(np.mean(auc_trains)),
        'auc_test': float(np.mean(auc_tests)),
        'ks_train': float(np.mean(ks_trains)),
        'ks_test': float(np.mean(ks_tests)),
        'psi': float(np.mean(psis)),
        'default': float(np.mean(defaults))
    }
    # 输出每套超参数最优模型
    best_model_idx = np.argmax(defaults)
    dump_pkl(models[best_model_idx])
    # 生成训练集测试集
    train = data.iloc[kf_list[best_model_idx][0]]
    write_data(train, 'train.fea')
    test = data.iloc[kf_list[best_model_idx][1]]
    write_data(test, 'test.fea')
    LOG.debug(metrics)
    nni.report_final_result(metrics)


if __name__ == '__main__':
    kf, data = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(kf, data, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
