from denoising.util import getGPU
import tensorflow as tf
from denoising.models.cnns.gfvnet import GFVNetModel
import os
import csv
from copy import deepcopy


# (1) change structure to mimic recon, (2) change direct denoising, (3) change number of steps per denoised geometry
def gftnrd_variations_configs():
    """Returns dictionary of gftnrd model configurations for training and testing"""

    configs = {

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

        '(1) GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=True,res_geometry_denoising=False,
                 steps_per_geometry=1, use_recon=True, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

        '(2) GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=False,res_geometry_denoising=False,
                 steps_per_geometry=1, use_recon=False, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

        '(3) GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=True,res_geometry_denoising=False,
                 steps_per_geometry=10, use_recon=False, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

        '(1,2) GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=False,res_geometry_denoising=False,
                 steps_per_geometry=1, use_recon=True, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

        '(1,3) GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=True,res_geometry_denoising=False,
                 steps_per_geometry=10, use_recon=True, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

        '(2,3) GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=False,res_geometry_denoising=False,
                 steps_per_geometry=10, use_recon=False, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

        '(1,2,3) GFTNRD': (
            GFVNetModel,
            dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                 num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                 test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                 constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                 scale_regularizer_weight=False, constant_regularizer_weight=False,
                 constant_geometry=False,
                 constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=False,res_geometry_denoising=False,
                 steps_per_geometry=10, use_recon=True, G={'name': 'TNRD',
                                                           'filters': 48,
                                                           'kernel_size': 7,
                                                           'grayscale': True}, F={'name': 'TNRD',
                                                                                  'filters': 48,
                                                                                  'kernel_size': 7,
                                                                                  'grayscale': True})),

    }
    return configs


def train_and_test_denoising_models():
    """Trains and test the denoising model architectures"""

    # Load model names and configs
    main_config = gftnrd_variations_configs()

    sigmas = [25]

    # Open csv file for writing
    save_file = os.path.join(os.getcwd(), 'reproduce/extra/gftnrd_variations', 'bsds68_gftnrd_variations_test.csv')
    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['Sigma'] + [str(sigma) for sigma in sigmas])

        for model_name, model_config in main_config.items():

            row = [model_name]

            for sigma in sigmas:
                model_config[1]['sigma'] = sigma
                denoising_model = model_config[0](deepcopy(model_config[1]), add_keys=False)
                row.append(denoising_model.test()['PSNR'])
                del denoising_model
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
            writer.writerow(row)


if __name__ == '__main__':
    getGPU()
    train_and_test_denoising_models()