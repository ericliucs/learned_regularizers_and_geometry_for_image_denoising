
from denoising.models.cnns.denoised_curvature_recon import DenoisedCurvatureReconModel


def base_config():
    """Returns base settings for curvature denoisers"""
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
        'F_train': True,
        'D_train': True,
        'S': 10,

    'D': {'epochs': 10,
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
    }


def configs():
    """Config settings that will be changed in each training"""
    return {

        'S': [10],

        'F': [

            {
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
                'scale_dataterm_weight': False,
                'constant_regularizer': False,
                'regularizer_weight': False,
                'scale_regularizer_weight': False,
                'use_prox': False,
                'S': 5,

                'R': {'name': 'TNRD',
                      'filters': 8,
                      'kernel_size': 3,
                      'grayscale': True}
            },

            {
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
                'scale_dataterm_weight': False,
                'constant_regularizer': False,
                'regularizer_weight': False,
                'scale_regularizer_weight': False,
                'use_prox': False,
                'S': 5,

                'R': {'name': 'TNRD',
                      'filters': 24,
                      'kernel_size': 5,
                      'grayscale': True}
            },

            {
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

        ]

    }


def generate_all_configs(configs):
    """Generates all configs based config settings that will be changed in each training."""
    prev_list_of_configs = []
    for key, items in configs.items():
        list_of_configs = []
        for item in items:
            if prev_list_of_configs:
                for config in prev_list_of_configs:
                    config_copy = config.copy()
                    list_of_configs.append({**config_copy, **{key: item}}.copy())
            else:
                list_of_configs.append({key: item})
        prev_list_of_configs = list_of_configs.copy()
    return prev_list_of_configs


def model_types_and_configs():
    """Returns a list of model types and configs that will be trained"""
    list_of_model_types_and_configs = []
    all_configs = generate_all_configs(configs())
    for config in all_configs:
        this_config = base_config()
        for key,value in config.items():
            this_config[key] = value
        list_of_model_types_and_configs.append((DenoisedCurvatureReconModel, this_config))
    for model_type, config in list_of_model_types_and_configs:
        config['D']['R']['kernel_size'] = config['F']['R']['kernel_size']
        config['D']['R']['filters'] = config['F']['R']['filters']
    return list_of_model_types_and_configs