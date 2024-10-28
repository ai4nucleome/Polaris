<img src="./doc/world_logo.jpg" alt="Polaris" title="Polaris" width="300">

# A Unified Axial-aware Framework for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data
**Polaris** is an advanced tool designed for rapid and precise detection of chromatin loops from various high-resolution contact maps, including Hi-C, single-cell Hi-C, Micro-C, DNA SPRITE and so on.

<div style="text-align: center;">
    <img src="./doc/Polaris.png" alt="Polaris Model" title="Polaris Model" width="600">
</div>


- Using examples for single cell Hi-C and bulk cell Hi-C loop annotations are under **example folder**.
- The scripts and data to **reproduce our analysis** can be found at: .

<b>NOTE:</b> We suggest users run Polaris on <b>GPU</b>. 
You can run Polaris on CPU for loop annotations, but it is much slower than on GPU. 

## Documentation
**Extensive documentation** can be found at:  .

## Installation
Polaris is developed and tested on Linux machines with python3.9 and relies on several libraries including pytorch, scipy, etc. 
We **strongly recommend** that you install Polaris in a virtual environment.

We suggest users using [conda](https://anaconda.org/) to create a virtual environment for it (It should also work without using conda, i.e. with pip). You can run the command snippets below to install Polaris:
<pre>
git clone https://github.com/BlanchetteLab/Polaris.git
cd Polaris
conda create -n polaris python=3.9
conda activate polaris
</pre>
Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website. It might be the following command depending on your cuda version:
<pre>
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
</pre>
Install additional library:
<pre>
pip install -r requirements.txt
</pre>
Install Polaris:
<pre>
pip install --editable .
</pre>
If fail, please try `python setup build` and `python setup install` first.

The installation requires network access to download libraries. Usually, the installation will finish within 5 minutes. The installation time is longer if network access is slow and/or unstable.

## Quick Start for Loop Annotation
<pre>
polaris loop pred -i [input mcool file] -o [output path for annotated loops]
</pre> 
It outputs predicted loops from the input contact map at 5kb resolution.
### output format
It contains tab separated fields as follows:
<pre>Chr1    Start1    End1    Chr2    Start2    End2    Score</pre>
|     Field     |                                  Detail                                 |
|:-------------:|:-----------------------------------------------------------------------:|
|   Chr1/Chr2   | chromosome names                                                        |
| Start1/Start2 | start genomic coordinates                                               |
|   End1/End2   | end genomic coordinates (i.e. End1=Start1+resol)                        |
|     Score     | Polaris's loop score [0~1]                                              | 


## Contact
A GitHub issue is preferable for all problems related to using Polaris. 

For other concerns, please email Yusen Hou or Yanlin Zhang (yhou925@connect.hkust-gz.edu.cn,  yanlinzhang@hkust-gz.edu.cn).