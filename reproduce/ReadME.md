# Reproduction

This directory contains code for reproducing the main results I have generated while working
in the area of image denoising. The code is
broken up into different types of results:

``test`` - For testing that the code-base is working correctly. Trains a simple
TNRD model.

``denoisers`` - During the past few years, I have implemented various old and new
variational based learning models for image denoising. This directory contains code 
that shows how to use my code-base to train these models and test them.

``paper`` - For generating the main results table from our BMVC 2021 paper on 
[Learned Regularizers and Geometry for Image Denoising](https://www.bmvc2021-virtualconference.com/assets/papers/1117.pdf).

``thesis`` - I wrote a master's thesis during the end of my time at Duquesne University. 
The thesis is focused on CNNs in population genetics and image denoising. This directory contains
the code used to generate the results for the second chapter of my thesis on image 
denoising. The thesis was embargoed and will be published most likely in August 2023.

``extra`` - Some extra experiments I conducted on the GFTNRD architecture based on 
the results within my thesis.

Each directory is composed of a code and results folder. The code generates the results
while the results folder stores the test values and any generated plots.