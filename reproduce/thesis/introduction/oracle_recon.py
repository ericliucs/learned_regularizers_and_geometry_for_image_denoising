import os
from PIL import Image
import numpy as np
from util import getGPU, psnr
import tensorflow as tf
import csv
from generator.curvature_layer import Curvature
from matplotlib import pyplot as plt


def save_image(img,name, save_dir):
    """

    Parameters
    ----------
    img : np image array
    name : name of image for saving
    save_dir : Dir to save image to
    change : Method of scaling image for saving
    """
    plt.clf()
    plt.imshow(np.abs(img), cmap='gray')
    #plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, name + '.png'),bbox_inches='tight')


def load_test_image():
    file = os.path.join('data', 'BSDS68', '3.png')
    with Image.open(file) as im:
        img = np.asarray(im)
        img = np.mean(img.astype('float32'), 2, keepdims=True) / 255.0
    return img[np.newaxis, :, :, :]


def descent_step(u_t_1, Ku, Ka, f, t=0.01, lambdaa=0.05):
    return u_t_1 + t*(Ku-Ka + lambdaa*(f-u_t_1))


def run_reconstruction(image, regularizer, save_dir, iterations = 500):

    clean = tf.constant(image, dtype=tf.float32)
    noise = tf.random.normal(shape=tf.shape(clean), mean=0, stddev=25/255)
    f = clean + noise
    save_image(f[0, :, :, 0], f'oracle_recon_noisy_image', save_dir)
    save_image(clean[0, :, :, 0], f'oracle_recon_clean_image', save_dir)


    print('Beginning Reconstruction')
    total_iter = 0
    x = f
    Ka = curvature(clean)
    save_image(Ka[0, :, :, 0], f'oracle_recon_Ka', save_dir)
    psnrs = []
    count = 0
    counts = [0]
    psnrs.append(psnr(x.numpy(), clean.numpy()))
    save_image(x.numpy()[0, :, :, 0], f'oracle_recon_iter_{0}', save_dir)
    for i in range(iterations):
        Ku = regularizer(x)
        x = descent_step(x, Ku, Ka, f)
        print(psnr(x.numpy(), clean.numpy()))
        count += 1
        counts.append(count)
        psnrs.append(psnr(x.numpy(), clean.numpy()))
    save_image(x[0, :, :, 0], f'oracle_recon_image', save_dir)

    # plot psnr values
    plt.clf()
    plt.plot(counts, psnrs)
    plt.ylabel('PSNR')
    plt.xlabel('Iteration')
    plt.savefig(os.path.join(save_dir, 'psnr_recon_graph.png'))

if __name__ == '__main__':
    # Maybe optimize patch so that it maximizes pointwise variation grad to see how low and high the operator will go
    # on data between 0 and 1
    getGPU()
    curvature = Curvature()
    image = load_test_image()
    save_dir = os.path.join(os.getcwd(), 'reproduce/thesis/introduction/figs')
    run_reconstruction(image, curvature, save_dir)