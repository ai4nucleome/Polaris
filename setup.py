"""
Setup script for Polaris.

A Universal Tool for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data.
"""

from setuptools import setup, find_packages

with open("README.md", "r") as readme:
    long_des = readme.read()

setup(
    name='polaris',
    version='1.0.0',
    author="Yusen HOU, Audrey Baguette, Mathieu Blanchette*, Yanlin Zhang*",
    author_email="yhou925@connect.hkust-gz.edu.cn",
    description="A Universal Tool for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data",
    long_description=long_des,
    long_description_content_type="text/markdown",
    url="https://github.com/ai4nucleome/Polaris",
    packages=['polaris'],
    include_package_data=True,
    install_requires=[
        'setuptools==75.1.0',
        'appdirs==1.4.4',
        'click==8.0.1',
        'cooler==0.8.11',
        'matplotlib==3.8.0',
        'numpy==1.22.4',
        'pandas==1.3.0',
        'scikit-learn==1.4.2',
        'scipy==1.7.3',
        'torch==2.2.2',
        'timm==0.6.12',
        'tqdm==4.65.0',
    ],
    entry_points={
        'console_scripts': [
            'polaris = polaris.polaris:cli',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
