import numpy as np
from scipy.ndimage import convolve
from reproduce.thesis.introduction.tv_denoising import load_test_image, save_image
import os



if __name__ == '__main__':
    difference_kernel = np.array([[0,0,0],[0,-1,0],[0,1,0]])
    image = load_test_image()[0,:,:,0]
    output = convolve(image, difference_kernel)

    save_dir = os.path.join(os.getcwd(), 'reproduce/thesis/introduction/figs')
    save_image(image, 'clean_image', save_dir, colobar=True)
    save_image(output, 'diffx_image', save_dir, colobar=True)
    save_image(difference_kernel, 'diffx_kernel', save_dir, absolute=False, colobar=True)

    difference_kernel = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    output = convolve(image, difference_kernel)
    save_image(difference_kernel, 'diffy_kernel', save_dir, absolute=False, colobar=True)
    save_image(output, 'diffy_image', save_dir, colobar=True)