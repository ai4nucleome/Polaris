"""
Setup script for Polaris.

A Unified Axial-aware Framework for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data
"""

from setuptools import setup, find_packages
with open("README.md","r") as readme:
    long_des=readme.read()

setup(
    name='polaris',
    version='1.0.0',
    author="Yusen HOU, Yanlin Zhang",
    author_email="yhou925@connect.hkust-gz.edu.cn",
    description="A Unified Axial-aware Framework for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data",
    long_description=long_des,
    long_description_content_type="text/markdown",
    url="https://github.com/compbiodsa/Polaris",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'polaris = polaris.polaris:cli',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Computational Biology',
    ],
    python_requires='>=3.9',
)



