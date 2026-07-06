# Conda Environments

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This folder documents the intended conda environments for the maintained
pipeline rather than providing machine-specific lock files.

本目录记录当前维护中的 pipeline 推荐 conda 环境名称与职责，不提供与具体机器强绑定的完整锁定文件。

## 中文

## 推荐环境名

- `fiji_env`
  - Fiji/ImageJ 启动和相关工具
- `caiman`
  - CaImAn motion correction，以及部分 preprocess / GUI 依赖
- `postmanual_analysis`
  - `06-18` 的后半段分析与绘图
- `suite2p`
  - suite2p ROI candidate generation

当前总入口默认就按这些环境名调用。

## 为什么这里只放轻量说明

- 机器相关的完整 lock file 很容易过时
- 不同平台的包解析结果也可能不同
- 这里更适合作为“环境命名契约”和“最小检查说明”

## 建议检查

```bash
conda env list
conda run -n fiji_env python -c "import imageio, tifffile, numpy"
conda run -n caiman python -c "import caiman, cv2, tifffile, numpy"
conda run -n postmanual_analysis python -c "import numpy, pandas, scipy, matplotlib, tifffile, sklearn, networkx, igraph, leidenalg, umap"
conda run -n suite2p python -c "import suite2p; print(suite2p.__version__)"
```

## English

## Recommended Environment Names

- `fiji_env`
  - Fiji/ImageJ launcher utilities
- `caiman`
  - CaImAn motion correction and several preprocess / GUI dependencies
- `postmanual_analysis`
  - post-manual analysis and plotting for steps `06-18`
- `suite2p`
  - suite2p ROI candidate generation

The main launcher currently expects these names by default.

## Why This Folder Stays Lightweight

- full machine-specific lock files drift quickly
- dependency resolution differs across platforms
- this folder is better used as a naming contract plus a minimal verification guide

## Suggested Checks

```bash
conda env list
conda run -n fiji_env python -c "import imageio, tifffile, numpy"
conda run -n caiman python -c "import caiman, cv2, tifffile, numpy"
conda run -n postmanual_analysis python -c "import numpy, pandas, scipy, matplotlib, tifffile, sklearn, networkx, igraph, leidenalg, umap"
conda run -n suite2p python -c "import suite2p; print(suite2p.__version__)"
```
