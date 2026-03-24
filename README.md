# Bleeding Masking Algorithm

This repository provides a demonstration Jupyter notebook for a bleeding-masking algorithm developed for KMTNet images (Shin et al., in preparation). The notebook shows how bleeding-affected pixels are identified and masked, and allows users to compare the generated masks with example KMTNet single-epoch images.

## Overview

CCD bleeding occurs when charge gets trapped in a bright pixel and slowly leaks out to subsequent pixels during CCD readout. These artifacts can affect source detection and photometry and therefore should be masked. This notebook provides a simple example of how the masking algorithm can be applied to KMTNet single-epoch data.

## Contents

- `bleed_github.ipynb`: demonstration notebook
- `bleed_masking.py`: masking function used in the notebook
- `data/`: example FITS images used as input

## Tested environment

This notebook was developed and tested in a standard scientific Python environment with the following packages:

- Python 3.10.9
- numpy 1.26.3
- matplotlib 3.9.2
- astropy 6.1.3
- Jupyter 7.2.2

## Usage

Open the notebook and run the cells sequentially:

```bash
jupyter notebook bleed_github.ipynb
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
