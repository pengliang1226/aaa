# encoding: utf-8
"""
@author: pengliang.zhao
@time: 2020/12/7 11:34
@file: setup.py
@desc: 
"""
from setuptools import find_packages, setup

NAME = 'model_module'
DESCRIPTION = 'all model func'
AUTHOR = 'pengliang.zhao'
REQUIRES_PYTHON = '>=3.7'
VERSION = '0.0.1'
REQUIRED = ['numpy==1.18.1', 'pandas==1.0.5', 'scikit-learn==0.23.1', 'xgboost==1.1.1', 'lightgbm==2.3.1',
            'scipy==1.4.1', 'statsmodels==0.10.1', 'mlxtend==0.17.3', 'nni==1.9', 'category-encoders==2.2.2']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True
)
