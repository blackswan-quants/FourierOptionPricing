from setuptools import setup, Extension
import pybind11
import sys

cpp_args = ['-std=c++14', '-O3'] if sys.platform != 'win32' else ['/std:c++14', '/O2']

ext_modules = [
    Extension(
        'cpp_pricer',
        sources=['fft_pricer.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='cpp_pricer',
    version='1.0.0',
    description='C++ implementation of the FFT option pricer',
    ext_modules=ext_modules,
)
