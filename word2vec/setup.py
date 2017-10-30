from distutils.core import setup
from Cython.Build import cythonize
setup(name="mWord2vec", ext_modules=cythonize("word2vec_inner.pyx"))
