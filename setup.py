from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kernel',
    ext_modules=[
        CUDAExtension('kernel', ['kernel.cu', 'bind.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)