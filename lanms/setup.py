from setuptools import setup
from setuptools import Extension

example_module = Extension(name='adaptor',  # 模块名称
                           sources=['adaptor.cpp', 'include/clipper/clipper.cpp'],    # 源码
                           # include_dirs=[r'include',     # 依赖的第三方库的头文件
                           #               r'D:\pybind11-master\include']
                           include_dirs=[r'include']
                           )

setup(ext_modules=[example_module])