# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/10/26 11:44
@file: EmbeddedMethod.py
@desc: 嵌入法，实际上是学习器自主选择特征。得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。
"""