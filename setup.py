"""
Setup script for Polaris.

A Unified Axial-aware Framework for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data
"""

from setuptools import setup, find_packages

setup(
    name='polaris',
    version='0.1.1',
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



