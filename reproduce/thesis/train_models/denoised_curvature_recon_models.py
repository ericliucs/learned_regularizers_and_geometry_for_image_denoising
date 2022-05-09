from reproduce.thesis.train_models.curvature_denoisers import all_curvature_denoisers
from reproduce.thesis.train_models.standard_denoisers import standard_denoiser_configs
from models.cnns.regularizers.regularizer_model import RegularizerModel
from models.cnns.denoised_curvature_recon import DenoisedCurvatureReconModel
from copy import deepcopy


def standard_denoised_curvature_recon_settings():
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
        'S': 5,
        'F_vnet_denoiser': True
    }


def denoised_curvature_recon_models():

    # Non boosting models
    new_list_configs = []
    for list_of_configs, model in all_curvature_denoisers():
        for config in list_of_configs:
            new_config = standard_denoised_curvature_recon_settings()
            new_config['F'] = config
            if model == RegularizerModel:
                new_config['F_vnet_denoiser'] = False
            new_list_configs.append(new_config)
    # Boosting models
    boosting_list = []
    for recon_config in new_list_configs:
        denoiser_configs, model = standard_denoiser_configs()
        for standard_denoiser_config in denoiser_configs:
            new_recon_config = deepcopy(recon_config)
            new_recon_config['D'] = standard_denoiser_config
            boosting_list.append(new_recon_config)
    return new_list_configs+boosting_list, DenoisedCurvatureReconModel