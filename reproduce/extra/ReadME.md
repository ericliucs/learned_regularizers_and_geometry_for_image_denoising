# Extra

This directory contains extra code I used to examine some variations of the 
GF-TNRD model based on the results of my thesis. On the BSDS68 dataset with sigma=25,
I found that adding the following properties to the model led to an increase in performance.

- For each denoised geometry, applying the regularization steps multiple times.
- Using an indirect denoising approach when applying each denoiser to each geometry.

Adding the above led to a .07DB increase in performance.

