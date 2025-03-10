<img src="./logo.png" alt="Polaris" title="Polaris" width="400">

# A Versatile Framework for Chromatin Loop Annotation in Bulk and Single-cell Hi-C Data

<a href="https://github.com/ai4nucleome/Polaris/releases/latest">
   <img src="https://img.shields.io/badge/Polaris-v1.1.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <!-- <img src="https://img.shields.io/badge/dependencies-tested-green"> -->
</a>  

🌟 **Polaris** 是一款高效的命令行工具，专为从多种实验（包括bulk Hi-C、scHi-C、Micro-C和DNA SPRITE）生成的接触图谱中快速准确检测染色质环而设计。该工具特别适合分析**稀疏的scHi-C数据和低覆盖度数据集**。

<div style="text-align: center;">
    <img src="./Polaris.png" alt="Polaris Model" title="Polaris Model" width="600">
</div>


• 单细胞 Hi-C 和 bulk Hi-C 的 loop 注释示例见 [**example文件夹**](https://github.com/ai4nucleome/Polaris/tree/master/example)

• 重现论文分析的脚本和数据见：[**Polaris Reproducibility**](https://zenodo.org/records/14294273)

> ❗️<b>重要提示❗️:</b> 建议用户在<b> GPU </b>上运行 Polaris。
> 虽然可以在 CPU 上运行，但速度会显著慢于 GPU。

> ❗️**重要提示❗️:** 如遇`CUDA OUT OF MEMORY`错误，请：
> - 检查 GPU 状态和可用显存
> - 减小 --batchsize 参数（默认值 128 需要约 36GB 显存，设置为 24 时显存需求将降至 10GB 以下）

## [📝文档](https://nucleome-polaris.readthedocs.io/en/latest/)
详细文档请访问：[Polaris 文档](https://nucleome-polaris.readthedocs.io/en/latest/)

## 安装
Polaris基于 python3.9 开发测试，建议使用 conda 创建虚拟环境：

```bash
git clone https://github.com/ai4nucleome/Polaris.git
cd Polaris
conda create -n polaris python=3.9
conda activate polaris
```
-------
安装Polaris：
```bash
./setup.sh
```
该脚本会自动从 [Hugging Face](https://huggingface.co/rr-ss/Polaris) 下载模型权重并完成安装。

也可手动下载[模型权重文件](https://huggingface.co/rr-ss/Polaris/resolve/main/polaris/model/sft_loop.pt?download=true)并放置于`Polaris/polaris/model`目录下重命名为 `sft_loop.pt`。

安装过程需要网络连接，正常情况下可在3分钟内完成。

## 快速开始
**查看详细参数说明：**
```bash
polaris --help
```
或访问[文档](https://nucleome-polaris.readthedocs.io/en/latest/)。

---
5kb分辨率下快速运行：
```bash
polaris loop pred -i [input.mcool] -o [output_path]
```
输出结果为预测的染色质环列表。

---
### 输出格式
制表符分隔的7列格式：
```
Chr1    Start1    End1    Chr2    Start2    End2    Score
```
|     列名     |                               说明                               |
|:-----------:|:---------------------------------------------------------------:|
|  Chr1/Chr2  | 染色体名称                                                       |
| Start1/Start2 | 起始基因组坐标                                                   |
|  End1/End2  | 终止基因组坐标（End1 = Start1+分辨率 * 1）                             |
|    Score    | Polaris预测分数 [0~1]                                            | 

## 引用
Yusen Hou, Audrey Baguette, Mathieu Blanchette*, & Yanlin Zhang*. __A versatile tool for chromatin loop annotation in bulk and single-cell Hi-C data__. _bioRxiv_, 2024. [Paper](https://doi.org/10.1101/2024.12.24.630215)
<br>
```
@article {Hou2024Polaris,
	title = {A versatile tool for chromatin loop annotation in bulk and single-cell Hi-C data},
	author = {Yusen Hou, Audrey Baguette, Mathieu Blanchette, and Yanlin Zhang},
	journal = {bioRxiv}
	year = {2024},
}
```

## 📩 联系我们
建议通过GitHub issues提交问题。

其他事宜请联系：Yusen Hou 或 Yanlin Zhang (yhou925@connect.hkust-gz.edu.cn,  yanlinzhang@hkust-gz.edu.cn)