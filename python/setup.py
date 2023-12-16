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

# pymicrovecdb_module = Extension(
#     'pymicrovecdb',
#     sources=['pymicrovecdb.cpp'],
#     include_dirs=[
#         os.path.join(BASE_DIR, 'include'),
#         os.path.join(BASE_DIR, 'include', 'index'),
#         os.path.join(BASE_DIR, 'faiss', 'include'),
#         os.path.join(BASE_DIR, 'fasttext','src')
#     ],
#     library_dirs=[os.path.join(BASE_DIR, 'lib'), os.path.join(BASE_DIR, 'faiss', 'lib')],
#     libraries=['microvecdb', 'rocksdb', 'faiss', 'faiss_c'],
#     extra_compile_args=['-std=c++17']
# )

setup(
    name='pymicrovecdb',
    version='0.1.0',
    description='C++ package',
    # ext_modules=[pymicrovecdb_module],
    package_dir={'': '.'},
    packages=['pymicrovecdb'],
    include_package_data=True,
    package_data={'pymicrovecdb': [
        os.path.join(BASE_DIR, 'lib', 'libmicrovecdb.so'),
        os.path.join(BASE_DIR, 'lib', 'microvecdb.cpython-310-x86_64-linux-gnu.so'),
        os.path.join(BASE_DIR, 'faiss', 'lib', 'libfaiss.so'),
        os.path.join(BASE_DIR, 'faiss', 'lib', 'libfaiss_c.so')
    ]}
    # data_files=[('lib', ['lib/libmicrovecdb.so', 'lib/libfaiss.so', 'lib/libfaiss_c.so'])]
)