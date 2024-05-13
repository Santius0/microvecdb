#!/usr/bin/env python3
# encoding: utf-8

# // run: python setup.py bdist_wheel sdist
# //      from inside the 'python' folder (a.k.a this folder)
# //      then do a: pip install .
# //      from in the 'python' folder

from distutils.core import setup, Extension
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

module = Extension("microvecdb",
                   sources = [f"{BASE_DIR}/python/pymicrovecdb.cpp"],
                   include_dirs=[f"{BASE_DIR}/include/core", f"{BASE_DIR}/include/index", f"{BASE_DIR}/include/operators",
                                 f"{BASE_DIR}/include/storage", f"{BASE_DIR}/faiss/include", f"{BASE_DIR}/annoy",
                                 f"{BASE_DIR}/SPTAG/AnnService", f"{BASE_DIR}/SPTAG/AnnService/inc",
                                 f"{BASE_DIR}/numpy/include"],
                   library_dirs=[f"{BASE_DIR}/lib", f"{BASE_DIR}/SPTAG/Release"],
                   libraries=['microvecdb', 'DistanceUtils', 'SPTAGLibStatic', 'pthread'],
                   extra_compile_args=['-std=c++17', '-fopenmp', '-fPIC']
                   )

setup(
    name='microvecdb',
    version='0.1.0',
    description='MicroVecDB C++ Interface Package',
    ext_modules=[module],
)