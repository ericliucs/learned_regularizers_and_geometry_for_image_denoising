# Experiment Results

While I wasn't able to complete much work on extending the software to include different noise models,
below are the PSNR results (in dB) on testing the code on different sigma levels comparing between the non-geometric 
and geometric frameworks. 

| sigma  | 5     | 35    | 60   | 75    |
|--------|-------|-------|------|-------|
| TNRD   | 37.80 | 27.46 | TODO | 24.51 |
| GFTNRD | 37.84 | 27.48 | TODO | 27.56 |

Image quality was analyzed by eye - image quality was consistent with the results of the paper, particularly the notes 
of section 4. The images are not included in the repo due to size, but can be reproduced through the code in the 
`reproduce` directory. These results are consistent with the findings of the paper, which provides more evidence in 
favor of the GF-TNRD model. In particular, the comparable performance and ability to preserve fine-features within 
images shows potential in providing a more translucent approach to ML-based denoising.
