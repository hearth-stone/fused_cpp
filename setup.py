"""Build script for fused_cpp C++ extension."""
import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

sources = sorted(glob.glob("csrc/*.cpp"))

setup(
    ext_modules=[
        CppExtension(
            name="fused_cpp._C",
            sources=sources,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
