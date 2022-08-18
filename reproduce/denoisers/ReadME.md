# Comparison of Denoising Models

This directory contains code that trains and test multiple types of variational denoising models including
TNRD, GF-TNRD, KB-TNRD, DnCNN, and TDV. The csv results for each model tested on the BSDS68 dataset are 
stored in the ``results`` directory. Please note that the trained TDV model has a different
training scheme than what was used in the original paper.


| Sigma | 15        | 25  | 50 |
|-------|-----------|-----|-------|
|TNRD| 31.45     | 28.95 | 26.00 |
|GFTNRD| 31.48     | 28.97 |  26.05 |
|KBTNRD| 31.51     | 29.01 | 26.09   |
|DnCNN| 31.72     | 29.22 | 26.23    |
|TDV| **31.82** | **29.36** | **26.43**  |



