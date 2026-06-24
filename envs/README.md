# Conda environments

This folder documents the intended environments for the maintained pipeline.

On the current Mac, existing useful environments are:

- `fiji_env`: Fiji/ImageJ launcher utilities and PyImageJ-related packages.
- `caiman`: CaImAn motion correction and general image-processing stack.
- `postmanual_analysis`: dedicated post-manual analysis and plotting stack for steps `06-18`.
- `suite2p`: suite2p `0.14.4`, preferred for this dataset.

The pipeline launcher uses these names by default where applicable. These files
are intentionally small, human-readable specs rather than full lock files.
Full lock files are machine-specific and can be regenerated later if needed.

Suggested checks:

```bash
conda env list
conda run -n fiji_env python -c "import imageio, tifffile, numpy"
conda run -n caiman python -c "import caiman, cv2, tifffile, numpy"
conda run -n postmanual_analysis python -c "import numpy, pandas, scipy, matplotlib, tifffile, sklearn, networkx, igraph, leidenalg, umap"
conda run -n suite2p python -c "import suite2p; print(suite2p.__version__)"
```
