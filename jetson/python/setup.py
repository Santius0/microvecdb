#!/usr/bin/env python3
# encoding: utf-8

# // run: python setup.py bdist_wheel sdist
# //      from inside the 'python' folder (a.k.a this folder)
# //      then do a: pip install .
# //      from in the 'python' folder

from setuptools import setup, Extension, find_packages
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)


setup(
    name='pymicrovecdb',
    version='0.1.0',
    description='MicroVecDB C++ Interface Package',
    package_dir={'': '.'},
    packages=['pymicrovecdb'],
    include_package_data=True,
    package_data={'pymicrovecdb': [
        os.path.join(BASE_DIR, 'lib', 'libmicrovecdb.so'),
        os.path.join(BASE_DIR, 'lib', 'microvecdb.cpython-38-x86_64-linux-gnu.so'),
    ]},
    install_requires=[
        'numpy>=1.24.4'
    ],
    zip_safe=True
)