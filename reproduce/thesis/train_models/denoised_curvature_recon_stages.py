from reproduce.thesis.train_models.denoised_curvature_recon_models import standard_denoised_curvature_recon_settings
from models.cnns.denoised_curvature_recon import DenoisedCurvatureReconModel
from copy import deepcopy

def denoising_model():
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

def F_denoiser():
    return {
        'epochs': 10,
        'loss_function': 'mean_sum_squared_error_curvature_loss',
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


def denoised_curvature_recon_stages_models():
    list_of_configs = []
    config = standard_denoised_curvature_recon_settings()
    config['F'] = F_denoiser()
    for S in [2, 5, 7, 10, 12, 15]:
        config['S'] = S
        new_config = deepcopy(config)
        list_of_configs.append(new_config)
        d_config = deepcopy(config)
        d_config['D'] = denoising_model()
        list_of_configs.append(d_config)
    return list_of_configs, DenoisedCurvatureReconModel





