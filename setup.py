import sys
from setuptools import setup

setup(
    name='wavelets_pytorch',
    version='0.1',
    author='Tom Runia',
    author_email='tomrunia@gmail.com',
    url='https://github.com/tomrunia/PyTorchWavelets',
    description='Wavelet Transform in PyTorch',
    long_description='Fast CPU/CUDA implementation of the Continuous Wavelet Transform in PyTorch.',
    license='MIT',
    packages=['wavelets_pytorch'],
    install_requires=[
        'torch>=0.4.0',
        'numpy>=1.14.1',
        'scipy>=1.0.0',
    ],
    python_requires='>2.7,' + ",".join(["!=3.%s.*" % i for i in range(6)]),
    extras_require={
        'examples':[
            'matplotlib>=2.0.0',
        ],
    },
    scripts=[]
)
