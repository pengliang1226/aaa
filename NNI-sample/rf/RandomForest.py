import logging

import nni
import pandas as pd
from cal_metrics.cal_metrics import calc_ks, calc_auc, calc_gini, calc_psi, dump_pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import PMMLPipeline

LOG = logging.getLogger('random-forest-classification')


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
    Get default hyper-parameters.

    :return:
    """
    params = {
        # 默认参数
        "n_jobs": -1,
        # 更新参数
        "criterion": "gini",
        "max_features": "auto",
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_estimators": 100,
        "random_state": 0
    }
    return params


def get_model(PARAMS):
    """
    Get model according to given parameters.

    :param PARAMS:
    :return:
    """
    estimator = RandomForestClassifier()
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
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    # 计算指标
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
