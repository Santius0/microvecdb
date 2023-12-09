from distutils.core import setup, Extension

module = Extension('_mvdb',
                   sources=['../include/mvdb_wrap.cxx'],
                   extra_compile_args=['-std=c++17'],
                   include_dirs=['../include', '../faiss/include', '../fasttext/src'])

setup(name='mvdb',
      version='0.1',
      author='Sergio Mathurin',
      description="""Python bindings for MicroVecDB using SWIG""",
      ext_modules=[module],
      py_modules=['mvdb'])
