from setuptools import find_packages
from distutils.core import setup

setup(
    name='g1_interaction',
    version='1.0.0',
    author='SailroLab',
    license="MIT",
    packages=find_packages(),
    description='End-to-end policy for contact-aware SMPLX human motion retargeting to G1 robot',
    install_requires=[
        'isaacgym',
        'rsl-rl',
        'matplotlib',
        'numpy',
        'torch',
        'smplx',
        'scipy',
        'trimesh',
        'pyrender'
    ]
)
