"""Variational Inference with sparse approximating densities"""

from setuptools import setup, find_packages

setup(name='ptvi',
      version='0.1',
      description='VI with sparse precision matrices',
      packages=['ptvi'],
      install_requires=[
          'torch', 'numpy', 'pandas', 'matplotlib'
      ])
