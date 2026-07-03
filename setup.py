"""Builds the C decoder kernels; all metadata lives in pyproject.toml."""

import sys

import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "ft8lib._ckernels",
            sources=["src/ft8lib/_ckernels.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=(
                ["/O2"] if sys.platform == "win32" else ["-O3"]
            ),
        )
    ]
)
