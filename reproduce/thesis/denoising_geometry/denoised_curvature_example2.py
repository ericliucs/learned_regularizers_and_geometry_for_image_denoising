from models.cnns.vnet import VNetModel
from generator.datagenerator import Curvature
from reproduce.thesis.introduction.tv_denoising import load_test_image
import os
import numpy as np
from util import getGPU
from matplotlib import pyplot as plt

def get_model():
    return {
        'epochs': 10,
        'loss_function': 'mean_sum_squared_error_loss',
        'num_training_images': 400,
        'patch_size': 64,
        'batch_size': 64,
        'grayscale': True,
        'test': 'BSDS68',
        'test_task': 'denoising',
        'train': 'BSDS400',
        'train_task': 'denoising',
        'sigma': 25,
        'training_type': 'standard',

        'constant_dataterm_weight': False,
        'constant_regularizer': False,
        'descent_weight': False,
        'scale_descent_weight': False,
        'scale_dataterm_weight': False,
        'use_prox': False,
        'S': 5,

        'R': {'name': 'TNRD',
              'filters': 48,
              'kernel_size': 7,
              'grayscale': True}
    }


if __name__ == '__main__':

    # Get model configurations
    tnrd_denoising_config = get_model()
    tnrd_curvature_denoising_config = tnrd_denoising_config.copy()
    tnrd_curvature_denoising_config['loss_function'] =  'mean_sum_squared_error_curvature_loss'

    # Save clean and noisy image; noisy psnr
    getGPU()
    curvature = Curvature()
    image = load_test_image(image_num=3)
    noisy = image + np.random.normal(scale=25 / 255, size=image.shape)
    clean_curvature = curvature(image)
    noisy_curvature = curvature(noisy).numpy()
    model = VNetModel(tnrd_curvature_denoising_config, add_keys=False).model
    denoised_curvature = curvature(model(noisy))
    save_dir = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/example_images')

    # Make plot
    fig, axs = plt.subplots(1,4, figsize=(11,5))

    # Add clean image
    axs[0].imshow(image[0,:,:,0], cmap='gray')
    axs[0].set_title(r'$a$')
    axs[0].set_axis_off()

    axs[1].imshow(np.abs(clean_curvature[0, :, :, 0]), cmap='gray')
    axs[1].set_title(r'$|\kappa(a)|$')
    axs[1].set_axis_off()

    axs[2].imshow(np.abs(noisy_curvature[0,:,:,0]), cmap='gray')
    axs[2].set_title(r'$|\kappa(f)|$')
    axs[2].set_axis_off()

    axs[3].imshow(np.abs(denoised_curvature[0,:,:,0]), cmap='gray')
    axs[3].set_title(r'$|\kappa(\mathcal{F}_L(f))|$')
    axs[3].set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'approx_curvature_example'))
    plt.clf()

