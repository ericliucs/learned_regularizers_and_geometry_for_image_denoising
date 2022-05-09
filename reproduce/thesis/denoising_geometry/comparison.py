from models.cnns.denoised_curvature_recon import DenoisedCurvatureReconModel
from generator.curvature_layer import Curvature
from reproduce.thesis.introduction.tv_denoising import load_test_image
import numpy as np
from util import getGPU
from models.cnns.vnet import VNetModel
from util import psnr

def denoising_curvature_recon_config():
    config = {}
    config['training_type'] = 'standard'
    config['epochs'] = 10
    config['loss_function'] = 'mean_sum_squared_error_loss'
    config['num_training_images'] = 400
    config['patch_size'] = 64
    config['batch_size'] = 64
    config['grayscale'] = True
    config['test'] = 'BSDS68'
    config['test_task'] = 'denoising'
    config['train'] = 'BSDS400'
    config['train_task'] = 'denoising'
    config['sigma'] = 25

    # VNet Process
    config['S'] = 12
    config['F_vnet_denoiser'] = True
    config['F_learn'] = True

    # # Regularizer
    config['F'] = {
        'epochs': 10,
        'loss_function': 'mean_sum_absolute_error_curvature_loss',
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
    return config

def tnrd_config():
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

def curvature_recon_config():
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
        'S': 5,
        'R': {
            'name': 'TV'}
    }


if __name__ == '__main__':
    getGPU()

    image = load_test_image()
    noisy = image + np.random.normal(scale=25 / 255, size=image.shape)

    curvature_layer = Curvature()

    noisy_curvature = curvature_layer(noisy).numpy()[0,:,:,0]
    clean_curvature = curvature_layer(image).numpy()[0,:,:,0]
    clean = image[0,:,:,0]

    #curvature_recon_model = ApproxCurvatureOracleReconModel(curvature_recon_config().copy())
    tnrd_model = VNetModel(tnrd_config(), add_keys=False)
    denoised_curvature_recon_model = DenoisedCurvatureReconModel(denoising_curvature_recon_config(), add_keys=False)

    curvature_denoiser = denoised_curvature_recon_model.F

    tnrd_denoised = tnrd_model.model(noisy).numpy()[0,:,:,0]
    print(f'TNRD PSNR: {psnr(tnrd_denoised, clean)}')

    output_curvature_denoiser = curvature_denoiser(noisy).numpy()[0,:,:,0]
    print(f'F PSNR: {psnr(output_curvature_denoiser, clean)}')

    denoised_curvature = curvature_layer(curvature_denoiser(noisy)).numpy()[0,:,:,0]
    curvature_tnrd_denoised = curvature_layer(tnrd_model.model(noisy)).numpy()[0,:,:,0]
    #
    # original_diff = noisy_curvature - clean_curvature
    # save_image(original_diff, 'orig_diff', '')
    # plt.clf()
    # plt.hist(original_diff)
    # plt.savefig('orig_hist.png')
    #
    #
    # diff = denoised_curvature - clean_curvature
    # save_image(diff, 'diff', '')
    # plt.clf()
    # plt.hist(diff)
    # plt.savefig('hist.png')
    #
    # print(np.corrcoef(np.abs(diff.flatten()), np.abs(clean_curvature.flatten())))


    # Compute psnr quantiles
    quantiles = [0.01,0.02, 0.05, 0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.92,0.95,0.975,0.99,1]
    psnrs = []
    curvature_max = np.max(clean_curvature)
    flattened_clean_curvature = clean_curvature.flatten()
    flattened_noisy_curvature = noisy_curvature.flatten()
    flattened_denoised_curvature = denoised_curvature.flatten()
    flattened_curvature_tnrd_denoised = curvature_tnrd_denoised.flatten()
    print('========================================================================')
    print('========================================================================')
    for q in quantiles:
        q_val = q*curvature_max
        print(q)
        print(flattened_noisy_curvature[np.abs(flattened_clean_curvature) > q_val].shape)
        print(psnr(flattened_noisy_curvature[np.abs(flattened_clean_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_clean_curvature) > q_val], range=4+2*np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_denoised_curvature[np.abs(flattened_clean_curvature) > q_val].shape)
        print(psnr(flattened_denoised_curvature[np.abs(flattened_clean_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_clean_curvature) > q_val], range=4 + 2 * np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_curvature_tnrd_denoised[np.abs(flattened_clean_curvature) > q_val].shape)
        print(psnr(flattened_curvature_tnrd_denoised[np.abs(flattened_clean_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_clean_curvature) > q_val], range=4 + 2 * np.sqrt(2)))

    print('========================================================================')
    print('========================================================================')

    for q in quantiles:
        q_val = q*curvature_max
        print(q)
        print(flattened_noisy_curvature[np.abs(flattened_noisy_curvature) > q_val].shape)
        print(psnr(flattened_noisy_curvature[np.abs(flattened_noisy_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_noisy_curvature) > q_val], range=4+2*np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_denoised_curvature[np.abs(flattened_noisy_curvature) > q_val].shape)
        print(psnr(flattened_denoised_curvature[np.abs(flattened_noisy_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_noisy_curvature) > q_val], range=4 + 2 * np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_curvature_tnrd_denoised[np.abs(flattened_noisy_curvature) > q_val].shape)
        print(psnr(flattened_curvature_tnrd_denoised[np.abs(flattened_noisy_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_noisy_curvature) > q_val], range=4 + 2 * np.sqrt(2)))

    print('========================================================================')
    print('========================================================================')

    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_noisy_curvature[np.abs(flattened_curvature_tnrd_denoised) > q_val].shape)
        print(psnr(flattened_noisy_curvature[np.abs(flattened_curvature_tnrd_denoised) > q_val],
                   flattened_clean_curvature[np.abs(flattened_curvature_tnrd_denoised) > q_val], range=4 + 2 * np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_denoised_curvature[np.abs(flattened_curvature_tnrd_denoised) > q_val].shape)
        print(psnr(flattened_denoised_curvature[np.abs(flattened_curvature_tnrd_denoised) > q_val],
                   flattened_clean_curvature[np.abs(flattened_curvature_tnrd_denoised) > q_val], range=4 + 2 * np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_curvature_tnrd_denoised[np.abs(flattened_curvature_tnrd_denoised) > q_val].shape)
        print(psnr(flattened_curvature_tnrd_denoised[np.abs(flattened_curvature_tnrd_denoised) > q_val],
                   flattened_clean_curvature[np.abs(flattened_curvature_tnrd_denoised) > q_val], range=4 + 2 * np.sqrt(2)))

    print('========================================================================')
    print('========================================================================')

    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_noisy_curvature[np.abs(flattened_denoised_curvature) > q_val].shape)
        print(psnr(flattened_noisy_curvature[np.abs(flattened_denoised_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_denoised_curvature) > q_val],
                   range=4 + 2 * np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_denoised_curvature[np.abs(flattened_denoised_curvature) > q_val].shape)
        print(psnr(flattened_denoised_curvature[np.abs(flattened_denoised_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_denoised_curvature) > q_val],
                   range=4 + 2 * np.sqrt(2)))

    print('==================================================================================================')
    for q in quantiles:
        q_val = q * curvature_max
        print(q)
        print(flattened_curvature_tnrd_denoised[np.abs(flattened_denoised_curvature) > q_val].shape)
        print(psnr(flattened_curvature_tnrd_denoised[np.abs(flattened_denoised_curvature) > q_val],
                   flattened_clean_curvature[np.abs(flattened_denoised_curvature) > q_val],
                   range=4 + 2 * np.sqrt(2)))





