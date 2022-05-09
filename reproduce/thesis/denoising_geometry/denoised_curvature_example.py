from models.cnns.denoised_curvature_recon import DenoisedCurvatureReconModel
from models.cnns.vnet import VNetModel
from generator.datagenerator import Curvature
from reproduce.thesis.introduction.tv_denoising import load_test_image, save_image
from reproduce.thesis.train_models.denoised_curvature_recon_models import standard_denoised_curvature_recon_settings
import os
import numpy as np
from util import getGPU
from util import psnr, rmse
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
    denoised_curvature_recon_config = standard_denoised_curvature_recon_settings()
    denoised_curvature_recon_config['D'] = tnrd_denoising_config.copy()
    denoised_curvature_recon_config['F'] = tnrd_curvature_denoising_config.copy()

    # Save clean and noisy image; noisy psnr
    getGPU()
    curvature = Curvature()
    image = load_test_image(image_num=44)
    noisy = image + np.random.normal(scale=25 / 255, size=image.shape)
    save_dir = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/example_images')

    # Save tnrd denoised imag with psnr
    model = VNetModel(tnrd_denoising_config, add_keys=False)
    model = model.model
    denoised_tnrd_image = model(noisy).numpy()

    # Denoising curvature model output
    model = VNetModel(tnrd_curvature_denoising_config, add_keys=False).model
    clean_output_of_model = model(image)
    noisy_ouput_of_model = model(noisy)
    #denoised_tnrd_curvature_image = curvature(noisy_ouput_of_model).numpy()

    # Save boosted reconstructed image
    model = DenoisedCurvatureReconModel(denoised_curvature_recon_config, add_keys=False)
    model = model.model
    denoised_recon_image = model(noisy).numpy()
    save_image(denoised_recon_image[0, :, :, 0], 'a4_noisy_denoised_by_recon', save_dir, absolute=False)

    # Make first plot
    fig, axs = plt.subplots(2,2)

    # Add clean image
    axs[0,0].imshow(image[0,:,:,0], cmap='gray')
    axs[0,0].set_title(r'Clean Image $a$')
    axs[0, 0].set_axis_off()

    axs[0,1].imshow(denoised_tnrd_image[0, :, :, 0], cmap='gray')
    axs[0,1].set_title(r'$\mathrm{TNRD}(f)$' + f', PSNR={psnr(denoised_tnrd_image,image):.2f}')
    axs[0, 1].set_axis_off()

    axs[1,0].imshow(noisy_ouput_of_model[0,:,:,0], cmap='gray')
    axs[1,0].set_title(r'$\mathcal{F}_L(f)$' + f', PSNR={psnr(noisy_ouput_of_model,image):.2f}')
    axs[1, 0].set_axis_off()

    axs[1,1].imshow(denoised_recon_image[0,:,:,0], cmap='gray')
    axs[1,1].set_title(r'$\mathrm{KB-TNRD}(f)$' + f', PSNR={psnr(denoised_recon_image,image):.2f}')
    axs[1, 1].set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'denoised_outputs'))
    plt.clf()

    image_curvature = curvature(image)
    denoised_tnrd_image_curvature = curvature(denoised_tnrd_image)
    noisy_output_of_model_curvature = curvature(noisy_ouput_of_model)
    denoised_recon_image_curvature = curvature(denoised_recon_image)

    fig, axs = plt.subplots(2, 2)
    # Plot absolute curvature images
    axs[0, 0].imshow(np.abs(image_curvature[0, :, :, 0]), cmap='gray')
    axs[0, 0].set_title(r'$|\kappa(a)|$')
    axs[0, 0].set_axis_off()

    axs[0, 1].imshow(np.abs(denoised_tnrd_image_curvature[0, :, :, 0]), cmap='gray')
    axs[0, 1].set_title(r'$|\kappa(\mathrm{TNRD}(f))|$' + f', RMSE={rmse(denoised_tnrd_image_curvature, image_curvature):.3f}')
    axs[0, 1].set_axis_off()

    axs[1, 0].imshow(np.abs(noisy_output_of_model_curvature[0, :, :, 0]), cmap='gray')
    axs[1, 0].set_title(r'$|\kappa(\mathcal{F}_L(f))|$' + f', RMSE={rmse(noisy_output_of_model_curvature, image_curvature):.3f}')
    axs[1, 0].set_axis_off()

    axs[1, 1].imshow(np.abs(denoised_recon_image_curvature[0, :, :, 0]), cmap='gray')
    axs[1, 1].set_title(r'$|\kappa(\mathrm{KB-TNRD}(f))|$' + f', RMSE={rmse(denoised_recon_image_curvature, image_curvature):.3f}')
    axs[1, 1].set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'curvature_outputs'))
    plt.clf()

    recon_tnrd_diff = np.abs(denoised_recon_image - image) - np.abs(denoised_tnrd_image - image)
    tnrd_recon_diff = -recon_tnrd_diff
    recon_tnrd_diff[recon_tnrd_diff < 0] = 0
    tnrd_recon_diff[tnrd_recon_diff < 0] = 0

    fig, axs = plt.subplots(2,1, figsize=(6,9))
    # Plot differences from clean image
    im = axs[0].imshow(recon_tnrd_diff[0,:,:,0], cmap='gray')
    #axs[0].set_title(r'\max(|\mathrm{KB-TNRD}(f) - a| - |TNRD(f) - a|, 0)')
    axs[0].set_axis_off()
    plt.colorbar(im, ax=axs[0],fraction=0.046, pad=0.04)

    im = axs[1].imshow(tnrd_recon_diff[0,:,:,0], cmap='gray')
    #axs[1].set_title(r'$\max(|\mathrm{TNRD}(f) - a| - |\mathrm{KB-TNRD}(f) - a|$, 0)')
    axs[1].set_axis_off()
    plt.colorbar(im, ax=axs[1],fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'differences'))




    # psnr_file = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/example_images/psnr_vals.csv')
    #
    # with open(psnr_file, 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #
    #     writer.writerow(['Image', 'PSNR'])
    #
    #     # Save clean and noisy image; noisy psnr
    #     getGPU()
    #     image = load_test_image()
    #     noisy = image + np.random.normal(scale=25/255, size=image.shape)
    #     save_dir = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/example_images')
    #     save_image(image[0, :, :, 0], 'a1_clean_image', save_dir, absolute=False)
    #     save_image(noisy[0,:,:,0], 'noisy_image', save_dir, absolute=False)
    #     writer.writerow(['f', f'{psnr(noisy, image):.2f}'])
    #
    #
    #     # Save noisy curvature
    #     curvature = Curvature()
    #     clean_curvature = curvature(image)
    #     noisy_curvature = curvature(noisy)
    #     save_image(noisy_curvature.numpy()[0, :, :, 0], 'noisy_image_curvature', save_dir, absolute=False)
    #     save_image(clean_curvature.numpy()[0, :, :, 0], 'b1_clean_image_curvature', save_dir, absolute=True)
    #
    #     # Save tnrd denoised imag with psnr
    #     model = VNetModel(tnrd_denoising_config, add_keys=False)
    #     model = model.model
    #     denoised_tnrd_image = model(noisy).numpy()
    #     save_image(denoised_tnrd_image[0,:,:,0], 'a2_noisy_denoised_by_tnrd', save_dir, absolute=False)
    #     save_image(curvature(denoised_tnrd_image)[0, :, :, 0], 'b2_noisy_denoised_by_tnrd_curvature', save_dir, absolute=True)
    #     writer.writerow(['tnrd_denoised', f'{psnr(denoised_tnrd_image, image):.2f}'])
    #
    #     # Save denoised curvature image
    #     model = VNetModel(tnrd_curvature_denoising_config, add_keys=False).model
    #     clean_output_of_model = model(image)
    #     noisy_ouput_of_model = model(noisy)
    #     denoised_tnrd_curvature_image = curvature(noisy_ouput_of_model).numpy()
    #     writer.writerow(['F_L(f)', f'{psnr(noisy_ouput_of_model, image):.2f}'])
    #     save_image(clean_output_of_model[0, :, :, 0], 'clean_indirect_curvature_denoiser_output', save_dir, absolute=True)
    #     save_image(noisy_ouput_of_model[0, :, :, 0], 'a3_noisy_indirect_curvature_denoiser_output', save_dir, absolute=True)
    #     save_image(denoised_tnrd_curvature_image[0, :, :, 0], 'b3_noisy_indirect_curvature_denoiser_output_curvature', save_dir, absolute=True)
    #
    #     # Save boosted reconstructed image
    #     model = DenoisedCurvatureReconModel(denoised_curvature_recon_config, add_keys=False)
    #     model = model.model
    #     denoised_recon_image = model(noisy).numpy()
    #     save_image(denoised_recon_image[0, :, :, 0], 'a4_noisy_denoised_by_recon', save_dir, absolute=False)
    #     save_image(curvature(denoised_recon_image)[0, :, :, 0], 'b4_noisy_denoised_by_recon_curvature', save_dir, absolute=True)
    #     writer.writerow(['recon_denoised', f'{psnr(denoised_recon_image, image):.2f}'])
    #
    #     recon_tnrd_diff = np.abs(denoised_recon_image - image) - np.abs(denoised_tnrd_image - image)
    #     recon_tnrd_diff[recon_tnrd_diff < 0] = 0
    #     tnrd_recon_diff = np.abs(denoised_tnrd_image - image) - np.abs(denoised_recon_image - image)
    #     tnrd_recon_diff[tnrd_recon_diff < 0] = 0
    #     save_image(recon_tnrd_diff[0, :, :, 0], 'c1_recon_tnrd_diff', save_dir, colorbar=True, absolute=False)
    #     save_image(tnrd_recon_diff[0, :, :, 0], 'c2_tnrd_recon_diff', save_dir, colorbar=True, absolute=False)
    #
    #     # k_recon_tnrd_diff = curvature(denoised_recon_image) - curvature(denoised_tnrd_image)
    #     # k_tnrd_recon_diff = curvature(denoised_tnrd_image) - curvature(denoised_recon_image)
    #     # save_image(k_recon_tnrd_diff[0, :, :, 0], 'd1_k_recon_tnrd_diff', save_dir, colorbar=True, absolute=False)
    #     # save_image(k_tnrd_recon_diff[0, :, :, 0], 'd2_k_tnrd_recon_diff', save_dir, colorbar=True, absolute=False)
    #     #
    #     # diff = (np.abs(denoised_tnrd_image - image)) - np.abs(denoised_recon_image - image)
    #     # save_image(diff[0, :, :, 0], 'diff', save_dir, colorbar=True, absolute=False)
    #     # diff = (np.abs(denoised_tnrd_image - image)) - np.abs(denoised_recon_image - image)
    #     # diff[diff < 0 ] = 0
    #     # save_image(diff[0,:,:,0], 'diff_poss', save_dir, colorbar=True, absolute=False)
    #     # diff = (np.abs(denoised_tnrd_image - image)) - np.abs(denoised_recon_image - image)
    #     # diff[diff > 0] = 0
    #     # diff = np.abs(diff)
    #     # save_image(diff[0, :, :, 0], 'diff_neg', save_dir, colorbar=True, absolute=False)
    #
    #
    #

