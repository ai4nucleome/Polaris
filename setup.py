"""
Setup script for Polaris.

A Unified Axial-aware Framework for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data
"""

from setuptools import setup, find_packages

setup(
    name='polaris',
    version='1.0.0',
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
)



