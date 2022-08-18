from denoising.util import getGPU
import tensorflow as tf
from denoising.models.cnns.gfvnet import GFVNetModel
from denoising.models.cnns.vnet import VNetModel
from denoising.models.cnns.denoised_curvature_recon import DenoisedCurvatureReconModel
import os
import csv
from copy import deepcopy


def denoising_model_configs():
    """Returns dictionary of denoising model configurations for training and testing"""

    configs = {

        'TNRD': (
            VNetModel,
            dict(epochs=10, loss_function='mean_sum_squared_error_loss', num_training_images=400, patch_size=50,
                 batch_size=64, grayscale=True, test='BSDS68', test_task='denoising', train='BSDS400',
                 train_task='denoising', sigma=25, training_type='standard', constant_dataterm_weight=False,
                 scale_dataterm_weight=False, constant_regularizer=False, regularizer_weight=False,
                 scale_regularizer_weight=False, use_prox=False, S=5, R={'name': 'TNRD',
                                                                         'filters': 48,
                                                                         'kernel_size': 7,
                                                                         'grayscale': True})),

        'GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=True, res_geometry_denoising=False,
                 steps_per_geometry=1, use_recon=False, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

        'KBTNRD': (DenoisedCurvatureReconModel,
                   {'epochs': 10, 'loss_function': 'mean_sum_squared_error_loss', 'num_training_images': 400,
                    'patch_size': 64, 'batch_size': 64, 'grayscale': True, 'test': 'BSDS68', 'test_task': 'denoising',
                    'train': 'BSDS400', 'train_task': 'denoising', 'sigma': 25, 'training_type': 'standard',
                    'F_train': True, 'D_train': True, 'S': 10,
                    'D': dict(epochs=10, loss_function='mean_sum_squared_error_loss', num_training_images=400,
                              patch_size=64, batch_size=64, grayscale=True, test='BSDS68', test_task='denoising',
                              train='BSDS400', train_task='denoising', sigma=25, training_type='standard',
                              constant_dataterm_weight=False, scale_dataterm_weight=False, constant_regularizer=False,
                              regularizer_weight=False, scale_regularizer_weight=False, use_prox=False, S=5,
                              R=dict(name='TNRD', filters=48, kernel_size=7, grayscale=True)),
                    'F': dict(epochs=10,
                              loss_function='mean_sum_squared_error_curvature_loss',
                              num_training_images=400,
                              patch_size=64,
                              batch_size=64,
                              grayscale=True,
                              test='BSDS68',
                              test_task='denoising',
                              train='BSDS400',
                              train_task='denoising',
                              sigma=25,
                              training_type='standard',
                              constant_dataterm_weight=False,
                              scale_dataterm_weight=False,
                              constant_regularizer=False,
                              regularizer_weight=False,
                              scale_regularizer_weight=False,
                              use_prox=False,
                              S=5, R=dict(
                            name='TNRD', filters=48, kernel_size=7, grayscale=True))}),

        'DnCNN': (
            VNetModel,
            dict(epochs=20, loss_function='mean_sum_squared_error_loss', num_training_images=400, patch_size=50,
                 batch_size=64, grayscale=True, test='BSDS68', test_task='denoising', train='BSDS400',
                 train_task='denoising', sigma=25, training_type='standard', constant_dataterm_weight=False,
                 scale_dataterm_weight=False, constant_regularizer=False, regularizer_weight=False,
                 scale_regularizer_weight=False, use_prox=False, S=1, R={'name': 'DnCNN',
                                                                         'filters': 64,
                                                                         'depth': 17,
                                                                         'grayscale': True})),

        'TDV': (
            VNetModel,
            dict(epochs=6, loss_function='mean_sum_squared_error_loss', num_training_images=400, patch_size=50,
                 batch_size=64, grayscale=True, test='BSDS68', test_task='denoising', train='BSDS400',
                 train_task='denoising', sigma=25, training_type='layer_wise', constant_dataterm_weight=True,
                 scale_dataterm_weight=True, constant_regularizer=True, regularizer_weight=True,
                 scale_regularizer_weight=True, use_prox=True, S=10, R={'name': 'TDV',
                                                                        'filters': 32,
                                                                        'num_scales': 3,
                                                                        'multiplier': 1,
                                                                        'num_mb': 3,
                                                                        'grayscale': True})),

    }
    return configs


def train_and_test_denoising_models():
    """Trains and test the denoising model architectures"""

    # Load model names and configs
    main_config = denoising_model_configs()

    sigmas = [15, 25, 50]

    # Open csv file for writing
    save_file = os.path.join(os.getcwd(), 'reproduce/denoisers/results', 'bsds68_denoisers_test.csv')
    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['Sigma'] + [str(sigma) for sigma in sigmas])

        for model_name, model_config in main_config.items():

            row = [model_name]

            for sigma in sigmas:
                model_config[1]['sigma'] = sigma
                if 'F' in model_config and 'sigma' in model_config['F']:
                    model_config['F']['sigma'] = sigma
                if 'D' in model_config and 'sigma' in model_config['D']:
                    model_config['D']['sigma'] = sigma
                denoising_model = model_config[0](deepcopy(model_config[1]), add_keys=False)
                psnr_val = denoising_model.test()['PSNR']
                print(denoising_model.model.summary())
                print(psnr_val)
                row.append(psnr_val)
                del denoising_model
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
            writer.writerow(row)


if __name__ == '__main__':
    getGPU()
    train_and_test_denoising_models()
