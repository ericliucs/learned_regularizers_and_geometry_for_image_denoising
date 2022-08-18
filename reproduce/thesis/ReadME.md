# Thesis

Contains the code used to generate results in my thesis. All the experiments
were based around approximating and reconstructing from the denoised curvature of
the image level lines.

``curvature/approximation`` - Testing capability of TNRD regularizer to approximate 
the curvature operator.

``curvature/boosting`` - Testing capability of curvature denoised by TNRD to boost the output
of a standard TNRD denoising model.

``curvature/denoising`` - Testing capability of full TNRD model and its individual regularizers
to denoise the curvature of images.

``curvature/kbtnrd`` - Testing performance of KB-TNRD as defined in 
thesis in comparison to TNRD and GF-TNRD models.

``curvature/recon`` - Tests capability of reconstructing from TNRD denoised curvature.
