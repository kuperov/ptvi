"""Variational Inference with sparse approximating densities"""

from setuptools import setup, find_packages

setup(
    name="ptvi",
    version="0.2",
    description="VI for forecasting",
    packages=find_packages(),
    install_requires=["torch", "numpy", "pandas", "matplotlib"],
)
