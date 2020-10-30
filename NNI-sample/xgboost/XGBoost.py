import logging

import nni
import pandas as pd
from cal_metrics.cal_metrics import calc_ks, calc_auc, calc_gini, calc_psi, dump_pkl
from xgboost import XGBClassifier
from sklearn2pmml import PMMLPipeline

LOG = logging.getLogger('xgboost-classification')


def load_data(train='./train.fea', test='./test.fea', label='flag_y'):
    """
    Load data from file system.

    :param train: train data and label
    :param test: test data and label
    :param label: y label name
    :return: dataframe representation
    """
    train_data = pd.read_feather(train)
    train_data.set_index('autoindex', inplace=True)
    test_data = pd.read_feather(test)
    test_data.set_index('autoindex', inplace=True)
    y_train = train_data[label]
    X_train = train_data.drop(columns=label)
    y_test = test_data[label]
    X_test = test_data.drop(columns=label)
    return X_train, X_test, y_train, y_test


def get_default_parameters():
    """
    Get model according to given parameters.

    :return:
    """
    params = {
        # 默认参数
        "objective": 'binary:logistic',
        "booster": 'gbtree',
        "n_jobs": -1,
        # 更新参数
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 0
    }
    return params


def get_model(PARAMS):
    """
    Get model according to given parameters.

    :param PARAMS:
    :return:
    """
    estimator = XGBClassifier()
    for k in PARAMS:
        if hasattr(estimator, k):
            setattr(estimator, k, PARAMS.get(k))
    pipeline = PMMLPipeline([('estimator', estimator)])
    return pipeline


def run(X_train, X_test, y_train, y_test, model):
    """
    Train model, predict on test set and get model performance.

    :param X_train: train data
    :param X_test:
    :param y_train: train label
    :param y_test: test label
    :param model: specific model
    :return: report final result to nni
    """
    # 训练
    model.fit(X_train, y_train,
              estimator__early_stopping_rounds=50,
              estimator__eval_set=[(X_test, y_test)], 
              estimator__eval_metric='auc')
    y_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    # 指标计算
    ks_train, ks_test = calc_ks(y_pred, y_train), calc_ks(y_test_pred, y_test)
    auc_train, auc_test = calc_auc(y_pred, y_train), calc_auc(y_test_pred, y_test)
    gini_train, gini_test = calc_gini(y_pred, y_train), calc_gini(y_test_pred, y_test)
    psi = calc_psi(y_pred, y_test_pred)
    # 整合结果
    metrics = {
        'gini_train': gini_train,
        'gini_test': gini_test,
        'auc_train': auc_train,
        'auc_test': auc_test,
        'ks_train': ks_train,
        'ks_test': ks_test,
        'psi': psi,
        'default': 1.8 * ks_test - 0.8 * abs(ks_train - ks_test)
    }
    dump_pkl(model)
    LOG.debug(metrics)
    nni.report_final_result(metrics)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
