from denoising.util import getGPU
from denoising.models.cnns.gfvnet import GFVNetModel
from denoising.models.cnns.vnet import VNetModel
import os
import csv


def get_tnrd_base_config():
    """Returns base config for TNRD model architecture and training"""

    config = {'epochs': 10,
              'loss_function': 'mean_sum_squared_error_loss',
              'num_training_images': 400,
              'patch_size': 50,
              'batch_size': 64,
              'grayscale': True,
              'test': 'BSDS68',
              'test_task': 'denoising',
              'train': 'BSDS400',
              'train_task': 'denoising',
              'sigma': 25,
              'training_type': 'standard',
              'constant_dataterm_weight': False,
              'scale_dataterm_weight': False,
              'constant_regularizer': False,
              'regularizer_weight': False,
              'scale_regularizer_weight': False,
              'use_prox': False,
              'S': 5,

              'R': {'name': 'TNRD',
                    'filters': 48,
                    'kernel_size': 7,
                    'grayscale': True}
              }
    return config


def get_gftnrd_base_config():
    """Returns base config for GfTNRD model architecture and training"""

    config = {'training_type': 'standard', 'epochs': 10, 'loss_function': 'mean_sum_squared_error_loss',
              'num_training_images': 400, 'patch_size': 50, 'batch_size': 64, 'grayscale': True, 'test': 'BSDS68',
              'test_task': 'denoising', 'train': 'BSDS400', 'train_task': 'denoising', 'sigma': 25,
              'constant_dataterm_weight': False, 'scale_dataterm_weight': False, 'regularizer_weight': False,
              'scale_regularizer_weight': False, 'constant_regularizer_weight': False, 'constant_geometry': False,
              'constant_denoiser': False, 'use_prox': False, 'S': 1, 'direct_geometry_denoising': True,
              'steps_per_geometry': 1, 'use_recon': False, 'G': {'name': 'TNRD',
                                                                 'filters': 8,
                                                                 'kernel_size': 3,
                                                                 'grayscale': True}, 'F': {'name': 'TNRD',
                                                                                           'filters': 8,
                                                                                           'kernel_size': 3,
                                                                                           'grayscale': True}}
    return config


def get_test_settings():
    """Returns different settings that will be tested"""
    return {
        'kernel_size_and_filter_nums': [(3, 8), (5, 24), (7, 48), (9, 80)],
        'sigmas': [15, 25, 50],
        'stages': [3, 5, 7]
    }


def train_and_test_gftnrd_models():
    """Trains and test the gftnrd model architectures"""

    # Load base config and test settings
    test_settings = get_test_settings()

    # Open csv file for writing
    save_file = os.path.join(os.getcwd(), 'reproduce/paper/results', 'bsds68_gftnrd_test.csv')
    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [None, None, 'sigma=15', None, None, None, 'sigma=25', None, None, None, 'sigma=50', None, None])

        for S in test_settings['stages']:
            writer.writerow([f'stage {S}', 'G 3x3', 'G 5x5', 'G 7x7',
                              'G 9x9', 'G 3x3', 'G 5x5', 'G 7x7', 'G 9x9', 'G 3x3', 'G 5x5', 'G 7x7', 'G 9x9'])
            for f_kernel_size, f_filter_num in test_settings['kernel_size_and_filter_nums']:

                # First do, TNRD
                if f_kernel_size == 3:
                    row = ['TNRD']
                    for sigma in test_settings['sigmas']:
                        for g_kernel_size, g_filter_num in test_settings['kernel_size_and_filter_nums']:
                            tnrd_config = get_tnrd_base_config()
                            tnrd_config['S'] = S
                            tnrd_config['sigma'] = sigma
                            tnrd_config['R']['filters'] = g_filter_num
                            tnrd_config['R']['kernel_size'] = g_kernel_size
                            tnrd_model = VNetModel(tnrd_config, add_keys=False)
                            row.append(tnrd_model.test()['PSNR'])
                    writer.writerow(row)

                # Now do gftnrd
                row = [f'F {f_kernel_size}']
                for sigma in test_settings['sigmas']:
                    for g_kernel_size, g_filter_num in test_settings['kernel_size_and_filter_nums']:
                        gftnrd_config = get_gftnrd_base_config()
                        gftnrd_config['S'] = S
                        gftnrd_config['sigma'] = sigma
                        gftnrd_config['G']['filters'] = g_filter_num
                        gftnrd_config['G']['kernel_size'] = g_kernel_size
                        gftnrd_config['F']['filters'] = f_filter_num
                        gftnrd_config['F']['kernel_size'] = f_kernel_size
                        gftnrd_model = GFVNetModel(gftnrd_config, add_keys=False)
                        row.append(gftnrd_model.test()['PSNR'])
                writer.writerow(row)


if __name__ == '__main__':
    getGPU()
    train_and_test_gftnrd_models()
