"""Build script for fused_mla_cpp C++ extension."""
import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

sources = sorted(glob.glob("csrc/*.cpp"))

setup(
    ext_modules=[
        CppExtension(
            name="fused_mla_cpp._C",
            sources=sources,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
