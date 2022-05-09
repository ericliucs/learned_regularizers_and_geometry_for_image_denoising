import os
from PIL import Image
import numpy as np
from util import getGPU, psnr
import tensorflow as tf
from generator.curvature_layer import Curvature
from matplotlib import pyplot as plt

def save_image(img,name, save_dir, absolute=True, colorbar=False):
    """

    Parameters
    ----------
    img : np image array
    name : name of image for saving
    save_dir : Dir to save image to
    change : Method of scaling image for saving
    """
    plt.clf()
    if absolute:
        img = np.abs(img)
    plt.imshow(img, cmap='gray')
    #plt.colorbar()
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    plt.savefig(os.path.join(save_dir, name + '.png'),bbox_inches='tight')


def load_test_image(image_num = 3):
    file = os.path.join('data', 'BSDS68', f'{image_num}.png')
    with Image.open(file) as im:
        img = np.asarray(im)
        img = np.mean(img.astype('float32'), 2, keepdims=True) / 255.0
    return img[np.newaxis, :, :, :]


def descent_step(u_t_1, Ku, f, t=0.0001, lambdaa=0.01):
    return u_t_1 + t*(Ku + lambdaa*(u_t_1-f))


def run_reconstruction(image, regularizer, save_dir, iterations = [1000, 1000, 1000]):
    psnr_file = os.path.join(save_dir, 'rof_psnr.txt')

    with open(psnr_file, 'w') as txt_file:

        clean = tf.constant(image, dtype=tf.float32)
        noise = tf.random.normal(shape=tf.shape(clean), mean=0, stddev=25/255)
        f = clean + noise
        save_image(f[0, :, :, 0], f'noisy_image', save_dir)

        txt_file.write(f'f: {psnr(f.numpy(), clean.numpy()):.2f}\n')


        print('Beginning Reconstruction')
        total_iter = 0
        x = f
        for iter in iterations:
            for i in range(iter):
                Ku = regularizer(x)
                x = descent_step(x, Ku, f)
                print(psnr(x.numpy(), clean.numpy()))
            total_iter += iter
            save_image(x.numpy()[0,:,:,0], f'tv_recon_iter_{total_iter}', save_dir)
            txt_file.write(f'recon_iter_{total_iter}: {psnr(x.numpy(), clean.numpy()):.2f}\n')


if __name__ == '__main__':
    # Maybe optimize patch so that it maximizes pointwise variation grad to see how low and high the operator will go
    # on data between 0 and 1
    getGPU()
    curvature = Curvature()
    image = load_test_image()
    save_dir = os.path.join(os.getcwd(), 'reproduce/thesis/introduction/figs')
    save_image(image[0, :,:,0], 'clean_image', save_dir)
    run_reconstruction(image, curvature, save_dir)
