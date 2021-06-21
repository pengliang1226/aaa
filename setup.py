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
REQUIRED = ['numpy', 'pandas', 'scikit-learn', 'lightgbm',
            'scipy', 'statsmodels', 'mlxtend', 'category-encoders']

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
