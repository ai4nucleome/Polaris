<img src="./doc/logo.png" alt="Polaris" title="Polaris" width="400">

# A Universal Tool for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data

<a href="https://github.com/ai4nucleome/Polaris/releases/latest">
   <img src="https://img.shields.io/badge/Polaris-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <!-- <img src="https://img.shields.io/badge/dependencies-tested-green"> -->
</a>  


**Polaris** is a universal and efficient command line tool tailored for rapid and accurate chromatin loop detectionfrom from contact maps generated by various assays, including bulk Hi-C, scHi-C, Micro-C, and DNA SPRITE. Polaris is particularly well-suited for analyzing **sparse scHi-C data and low-coverage datasets**.

<div style="text-align: center;">
    <img src="./doc/Polaris.png" alt="Polaris Model" title="Polaris Model" width="600">
</div>


- Using examples for single cell Hi-C and bulk cell Hi-C loop annotations are under [**example folder**](https://github.com/ai4nucleome/Polaris/tree/master/example).
- The scripts and data to **reproduce our analysis** can be found at: [**Polaris Reproducibility**](https://zenodo.org/records/14294273).

> <b>NOTE:</b> We suggest users run Polaris on <b>GPU</b>. 
> You can run Polaris on CPU for loop annotations, but it is much slower than on GPU. 

> **Note:** If you encounter a `CUDA OUT OF MEMORY` error, please:
> - Check your GPU's status and available memory.
> - Reduce the --batchsize parameter. (The default value of 128 requires approximately 36GB of CUDA memory. Setting it to 24 will reduce the requirement to less than 10GB.)

## Documentation
**Extensive documentation** can be found at: [Polaris Doc](https://nucleome-polaris.readthedocs.io/en/latest/).

## Installation
Polaris is developed and tested on Linux machines with python3.9 and relies on several libraries including pytorch, scipy, etc. 
We **strongly recommend** that you install Polaris in a virtual environment.

We suggest users using [conda](https://anaconda.org/) to create a virtual environment for it (It should also work without using conda, i.e. with pip). You can run the command snippets below to install Polaris:

```bash
git clone https://github.com/ai4nucleome/Polaris.git
cd Polaris
conda create -n polaris python=3.9
conda activate polaris
```
Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website. It might be the following command depending on your cuda version:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```
Install Polaris:
```bash
pip install --use-pep517 --editable .
```
If fail, please try `python setup build` and `python setup install` first.

The installation requires network access to download libraries. Usually, the installation will finish within 5 minutes. The installation time is longer if network access is slow and/or unstable.

## Quick Start for Loop Annotation
```bash
polaris loop pred -i [input mcool file] -o [output path of annotated loops]
```
It outputs predicted loops from the input contact map at 5kb resolution.
### output format
It contains tab separated fields as follows:
```
Chr1    Start1    End1    Chr2    Start2    End2    Score
```
|     Field     |                                  Detail                                 |
|:-------------:|:-----------------------------------------------------------------------:|
|   Chr1/Chr2   | chromosome names                                                        |
| Start1/Start2 | start genomic coordinates                                               |
|   End1/End2   | end genomic coordinates (i.e. End1=Start1+resol)                        |
|     Score     | Polaris's loop score [0~1]                                              | 


## Citation:
Yusen Hou, Audrey Baguette, Mathieu Blanchette*, & Yanlin Zhang*. __A universal tool for chromatin loop annotation in bulk and single-cell Hi-C data__. _bioRxiv_, 2024. [Paper](https://doi.org/10.1101/2024.12.24.630215)
<br>
```
@article {Hou2024Polaris,
	title = {A universal tool for chromatin loop annotation in bulk and single-cell Hi-C data},
	author = {Yusen Hou, Audrey Baguette, Mathieu Blanchette, and Yanlin Zhang},
	journal = {bioRxiv}
	year = {2024},
}
```

## Contact
A GitHub issue is preferable for all problems related to using Polaris. 

For other concerns, please email Yusen Hou or Yanlin Zhang (yhou925@connect.hkust-gz.edu.cn,  yanlinzhang@hkust-gz.edu.cn).