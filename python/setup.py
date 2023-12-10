#!/usr/bin/env python3
# encoding: utf-8

from setuptools import setup, find_packages, Extension
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

microvecdb = Extension(
    'microvecdb',
    sources=['microvecdb_py.cpp'],
    include_dirs=[
        os.path.join(BASE_DIR, 'include'),
        os.path.join(BASE_DIR, 'faiss', 'include'),
        os.path.join(BASE_DIR, 'fasttext','src')
    ],
    library_dirs=[
        os.path.join(BASE_DIR, 'lib'),
        os.path.join(BASE_DIR, 'faiss', 'lib')
    ],
    libraries=['microvecdb', 'rocksdb', 'faiss', 'faiss_c'],
    extra_compile_args=['-std=c++17']
)

setup(
    name='microvecdb',
    version='0.1.0',
    data_files=[
        ('lib', [os.path.join(BASE_DIR, 'lib','libmicrovecdb.so')])
    ],
    description='Hello world module written in C',
    ext_modules=[microvecdb]
)