from models.cnns.approx_curvature_oracle_recon import ApproxCurvatureOracleReconModel
from reproduce.thesis.introduction.tv_denoising import load_test_image, save_image
import os
from util import getGPU
from models.cnns.regularizers.regularizer_model import RegularizerModel
import numpy as np

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
        'S': 5,
        'R': {
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
         'kernel_size': 7,
         'grayscale': True}
    },
    }

if __name__ == '__main__':
    getGPU()
    tnrd_approx_recon_model_config = standard_approx_curvature_recon_settings()
    tv_approx_recon_model_config = standard_approx_curvature_recon_settings()
    tv_approx_recon_model_config['R'] = {}
    tv_approx_recon_model_config['R']['name'] = 'TV'

    image = load_test_image()
    regularizer_tnrd  = RegularizerModel(tnrd_approx_recon_model_config['R'].copy(), add_keys=False).R
    R_tnrd = ApproxCurvatureOracleReconModel(tnrd_approx_recon_model_config.copy(), add_keys=False).R
    R_tv = ApproxCurvatureOracleReconModel(tv_approx_recon_model_config.copy(), add_keys=False).R

    ka = R_tv(image).numpy()[0,:,:,0]
    tnrd_approx = R_tnrd(image).numpy()[0,:,:,0]
    tnrd_r_approx = regularizer_tnrd(image).numpy()[0,:,:,0]

    save_image(ka, 'curvature', save_dir=os.getcwd())
    save_image(tnrd_approx, 'tnrd_approx', save_dir=os.getcwd())
    save_image(tnrd_r_approx, 'tnrd_r_approx', save_dir=os.getcwd())

    print(np.sum(np.abs(tnrd_approx-tnrd_r_approx)))