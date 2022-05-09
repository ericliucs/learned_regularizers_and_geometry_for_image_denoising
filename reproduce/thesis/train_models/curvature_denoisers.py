from reproduce.thesis.train_models.curvature_approx_models import full_curvature_approx_models
from reproduce.thesis.train_models.standard_denoisers import standard_denoiser_configs


def regularizer_curvature_denoisers():
    approx_models, model = full_curvature_approx_models()
    for config in approx_models:
        config['train_task'] = 'denoise_curvature'
        config['test_task'] = 'denoise_curvature'
    return approx_models, model


def full_curvature_denoisers():
    denoiser_models, model = standard_denoiser_configs()
    for config in denoiser_models:
        config['train_task'] = 'denoise_curvature'
        config['test_task'] = 'denoise_curvature'
    return denoiser_models, model


def regularizer_curvature_denoisers_image_input():
    approx_models, model = full_curvature_approx_models()
    for config in approx_models:
        config['train_task'] = 'denoising'
        config['test_task'] = 'denoising'
        config['loss_function'] = 'mean_sum_squared_error_curvature_loss'
    return approx_models, model


def full_curvature_denoisers_image_input():
    denoiser_models, model = standard_denoiser_configs()
    for config in denoiser_models:
        config['train_task'] = 'denoising'
        config['test_task'] = 'denoising'
        config['loss_function'] = 'mean_sum_squared_error_curvature_loss'
    return denoiser_models, model


def all_curvature_denoisers():
    return [regularizer_curvature_denoisers(), full_curvature_denoisers(), regularizer_curvature_denoisers_image_input(), full_curvature_denoisers_image_input()]
