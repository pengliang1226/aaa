# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/21 19:34
@file: test.py
@desc: 
"""
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from eval_metrics import ScoreStretch, calc_ks, calc_auc
from feature_binning import DecisionTreeBinner, QuantileBinner
from feature_select import nan_filter, unique_filter, mode_filter, PSI_filter, correlation_filter, vif_filter, logit_pvalue_forward_filter, \
    coef_forward_filter
from util import get_attr_by_unique, disorder_mapping, woe_transform

if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv(r'D:\workbook\jupyter notebook\金融风控\建模代码\data_pos.csv')

    ori_data = data.copy()
    ori_data.drop(columns='cus_num', inplace=True)
    drop_col = [v for v in ori_data.columns if re.match('flag_|score', v)]
    ori_data.drop(columns=drop_col, inplace=True)
    y = ori_data.pop('y')
    user_date = ori_data.pop('user_date')
    ori_data.insert(0, 'y', y)
    ori_data.insert(1, 'user_date', user_date)

    # 填充空值为-999
    ori_data.replace({'-111': -111, '-999': -999}, inplace=True)
    null_flag = {x: [-111, -999] for x in ori_data.columns[2:]}

    # 切分数据
    train_data = ori_data[~ori_data['user_date'].str.contains('2018-08')].copy()
    test_data = ori_data[ori_data['user_date'].str.contains('2018-08')].copy()

    # 根据缺失率，同值占比，唯一值占比进行筛选
    tmp = nan_filter(ori_data.iloc[:, 2:], null_flag=null_flag)
    tmp = mode_filter(ori_data.loc[:, tmp], null_flag=null_flag)
    first_feats = unique_filter(ori_data.loc[:, tmp], null_flag=null_flag)

    # 获取变量属性类型
    features_type = {}
    for col in first_feats:
        col_data = ori_data[col]
        features_type[col] = get_attr_by_unique(col_data, null_value=null_flag[col])

    # ----------------------------------------------PSI筛选----------------------------------------------#
    QT = QuantileBinner(features_info=features_type, features_nan_value=null_flag, max_leaf_nodes=6)
    QT.fit(train_data.loc[:, first_feats], train_data['y'])

    res = PSI_filter(train_data[train_data['user_date'] < '2018-04-30'],
                     train_data[train_data['user_date'] >= '2018-04-30'], bins_info=QT.features_bins,
                     feature_type=features_type)

    second_feats = list(res.keys())

    # ----------------------------------------------iv值筛选--------------------------------------------------#
    DT = DecisionTreeBinner(features_info=features_type, features_nan_value=null_flag, max_leaf_nodes=6)
    DT.fit(train_data.loc[:, second_feats], train_data['y'])

    special = []
    # 剔除分箱区间数为1的变量
    for k, v in DT.features_bins.items():
        if (v['flag'] == 1 and len(v['bins']) == 2) or (v['flag'] == 0 and len(v['bins']) == 1):
            special.append(k)

    # 剔除IV值小于0.02变量
    for k, v in DT.features_iv.items():
        if v <= 0.02:
            special.append(k)

    third_feats = [v[0] for v in sorted(DT.features_iv.items(), key=lambda x: x[1], reverse=True) if
                   v[0] not in special]

    # ----------------------------------------------相关系数筛选----------------------------------------------#
    forth_data = train_data.copy()
    for col in third_feats:
        if forth_data[col].dtype == 'O':
            col_data = forth_data[col]
            tmp = disorder_mapping(col_data, forth_data['y'], 1, null_flag[col])
            forth_data[col] = col_data.map(tmp).fillna(-1)
    forth_feats = correlation_filter(forth_data, third_feats)

    # ----------------------------------------------多重共线性筛选----------------------------------------------#
    fifth_feats = vif_filter(forth_data, forth_feats, threshold=10)

    # ----------------------------------------------调整分箱并重新筛选iv----------------------------------------------#
    DT.binning_trim(train_data.loc[:, fifth_feats], train_data['y'], fifth_feats)
    six_feats = [k for k in fifth_feats if DT.features_iv[k] > 0.02]

    # -----------------------------------------------woe转码---------------------------------------------------#
    train_data = train_data.loc[:, ['y', 'user_date'] + six_feats]
    test_data = test_data.loc[:, ['y', 'user_date'] + six_feats]
    for col in six_feats:
        train_data[col] = woe_transform(train_data[col], features_type[col], DT.features_bins[col],
                                        DT.features_woes[col])
        test_data[col] = woe_transform(test_data[col], features_type[col], DT.features_bins[col], DT.features_woes[col])

    # ---------------------------------------------------剔除系数为负数的特征-----------------------------------------#
    seven_feats = coef_forward_filter(train_data, train_data['y'], six_feats)

    # ------------------------------------------------显著性筛选------------------------------------------------------#
    eight_feats = logit_pvalue_forward_filter(train_data, train_data['y'], seven_feats)

    # -----------------------------------------训练模型------------------------------------------#
    train_x, val_x, train_y, val_y = train_test_split(train_data[eight_feats], train_data['y'], test_size=0.2,
                                                      random_state=123)
    test_x, test_y = test_data[eight_feats], test_data['y']
    params = {
        # 默认参数
        "solver": 'liblinear',
        "multi_class": 'ovr',
        # 更新参数
        "max_iter": 100,
        "penalty": "l2",
        "C": 1.0,
        "random_state": 0
    }
    lr_model = LogisticRegression(**params).fit(train_x, train_y)

    sc = ScoreStretch(S=ori_data['y'].sum() / ori_data.shape[0])
    train_pred = sc.predict(train_x, lr_model)
    train_score = sc.predict_transform(train_x, lr_model)
    val_pred = sc.predict(val_x, lr_model)
    val_score = sc.predict_transform(val_x, lr_model)
    test_pred = sc.predict(test_x, lr_model)
    test_score = sc.predict_transform(test_x, lr_model)

    val_ks = calc_ks(val_y, val_pred)
    test_ks = calc_ks(test_y, test_pred)

    val_auc = calc_auc(val_y, val_pred)
    test_auc = calc_auc(test_y, test_pred)
    print('验证集auc={}, ks={}'.format(val_auc, val_ks))
    print('测试集auc={}, ks={}'.format(test_auc, test_ks))