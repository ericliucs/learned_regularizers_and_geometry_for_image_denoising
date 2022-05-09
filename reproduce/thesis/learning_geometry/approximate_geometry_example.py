from models.cnns.regularizers.regularizer_model import RegularizerModel
from generator.datagenerator import Curvature
from reproduce.thesis.introduction.tv_denoising import load_test_image

import os
import numpy as np
from util import getGPU
from matplotlib import pyplot as plt


def standard_approx_curvature_recon_settings():
    return {
        'epochs': 10,
        'loss_function': 'mean_sum_squared_error_loss',
        'num_training_images': 400,
        'patch_size': 64,
        'batch_size': 64,
        'grayscale': True,
        'test': 'BSDS68',
        'test_task': 'oracle_recon',
        'train': 'BSDS400',
        'train_task': 'oracle_recon',
        'sigma': 25,
        'training_type': 'standard',
        'S': 5
    }


def get_model():
    return {
            'epochs': 10,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 64,
            'batch_size': 64,
            'grayscale': True,
            'test': 'BSDS68',
            'test_task': 'approx_curvature',
            'train': 'BSDS400',
            'train_task': 'approx_curvature',
            'sigma': 25,
            'training_type': 'standard',

            'R': {'name': 'TNRD',
         'filters': 48,
         'kernel_size': 3,
         'grayscale': True},

    }


if __name__ == '__main__':

    # Get model configurations
    tnrd_approx_model_config = get_model()
    ktnrd_approx_model_config = get_model()
    ktnrd_approx_model_config['R'] = {'name': 'KTNRD',
                  'filters': 48,
                  'kernel_size': 3,
                  'type': 'standard',
                  'grayscale': True}

    # Save clean image
    getGPU()
    image = load_test_image(image_num=3)
    noisy = image + np.random.normal(scale=25 / 255, size=image.shape)
    save_dir = os.path.join(os.getcwd(), 'reproduce/thesis/learning_geometry/results/example_images')
    #save_image(image[0, :, :, 0], 'clean_image', save_dir)

    # Plot a, k(a), gtnrd(a), gktnrd(a), absolute value
    model = RegularizerModel(tnrd_approx_model_config.copy(), add_keys=False).model
    tnrd_approx = model(image).numpy()

    model = RegularizerModel(ktnrd_approx_model_config.copy(), add_keys=False).model
    ktnrd_approx = model(image).numpy()

    curvature = Curvature()(image).numpy()

    # # Plot reconstructed estimates
    # recon_settings = standard_approx_curvature_recon_settings()
    # recon_settings['R'] = tnrd_approx_model_config
    # model = ApproxCurvatureOracleReconModel(recon_settings, add_keys=False).model
    # tnrd_approx_recon = model([noisy, image]).numpy()
    #
    # recon_settings = standard_approx_curvature_recon_settings()
    # recon_settings['R'] = ktnrd_approx_model_config
    # model = ApproxCurvatureOracleReconModel(recon_settings, add_keys=False).model
    # ktnrd_approx_recon = model([noisy, image]).numpy()

    #########################################################################################
    # Make first plot
    fig, axs = plt.subplots(1,3, figsize=(9,4))

    # Add clean image
    axs[0].imshow(image[0, :, :, 0], cmap='gray')
    axs[0].set_title(r'Clean Image $a$')
    axs[0].set_axis_off()

    axs[1].imshow(np.abs(curvature[0, :, :, 0]), cmap='gray')
    axs[1].set_title(r'$|\kappa(a)|$')
    axs[1].set_axis_off()

    axs[2].imshow(np.abs(tnrd_approx[0, :, :, 0]), cmap='gray')
    axs[2].set_title(r'$|\mathcal{G}_{L_1}(a)|$')
    axs[2].set_axis_off()

    print('saving')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gtnrd_approx_images'))
    plt.clf()

    # ##################################################################################
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(np.abs(tnrd_approx[0, :, :, 0] - curvature[0,:,:,0]), cmap='gray')
    # axs[0].set_title(r'$\mathcal{G}_{L_1}$' + f', RMSE={rmse(tnrd_approx, curvature):.3f}')
    # axs[0].set_axis_off()
    #
    # axs[1].imshow(np.abs(ktnrd_approx[0, :, :, 0] - curvature[0,:,:,0]), cmap='gray')
    # axs[1].set_title(r'$\mathcal{G}_{L_2}$' + f', RMSE={rmse(ktnrd_approx, curvature):.3f}')
    # axs[1].set_axis_off()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'differences'))
    # plt.clf()
    #
    # #######################################################################################
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(tnrd_approx_recon[0, :, :, 0], cmap='gray')
    # axs[0].set_title(r'$\mathcal{G}_{L_1}$' + f', PSNR={psnr(tnrd_approx_recon, image):.3f}')
    # axs[0].set_axis_off()
    #
    # axs[1].imshow(ktnrd_approx_recon[0, :, :, 0], cmap='gray')
    # axs[1].set_title(r'$\mathcal{G}_{L_2}$' + f', PSNR={psnr(ktnrd_approx_recon, image):.3f}')
    # axs[1].set_axis_off()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'recons'))
    # plt.clf()
    #


    #
    #
    # # Save clean curvature
    # curvature = Curvature()
    # clean_curvature = curvature(image)
    # save_image(clean_curvature.numpy()[0, :, :, 0], 'clean_curvature', save_dir)
    #
    # # Save tnrd approximation of clean curvature
    # model = RegularizerModel(approx_model_config, add_keys=False).model
    # approximation = model(image).numpy()
    # save_image(approximation[0,:,:,0], 'curvature_approx_image', save_dir)