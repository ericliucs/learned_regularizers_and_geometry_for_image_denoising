from denoising.models.cnns.approx_curvature_oracle_recon import ApproxCurvatureOracleReconModel
from reproduce.thesis.curvature.approximation.code.approx_curvature import approx_curvature_regularizer_models, test_and_save_results
import os


def recon_approx_curvature_models():
    """Returns list of model types and configurations that will be tested. Each model type and configuration corresponds
    to reconstruction

    Returns
    -------
    List of tuples of size 2. First entry of tuple is model type. Second entry of tuple is
        configuration of the model.

    """

    standard_recon_approx_curvature_config = {
        'epochs': 2,
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
    }
    list_of_model_types_and_configs = []
    stages = [2, 3, 5, 7, 10, 12, 15]
    for s in stages:
        # Add TV
        config = standard_recon_approx_curvature_config.copy()
        config['S'] = s
        config['R'] = {'name': 'TV'}
        list_of_model_types_and_configs.append((ApproxCurvatureOracleReconModel, config))
        # Add the rest
        for model_type, approx_curvature_config in approx_curvature_regularizer_models():
            config = standard_recon_approx_curvature_config.copy()
            config['S'] = s
            config['R'] = approx_curvature_config.copy()
            list_of_model_types_and_configs.append((ApproxCurvatureOracleReconModel, config))
    return list_of_model_types_and_configs


if __name__ == '__main__':
    test_and_save_results(recon_approx_curvature_models(),
                       os.path.join(os.getcwd(),
                        'reproduce/thesis/curvature/approximation/results/spreadsheets/recon_approx_curvature.csv'),
                          col_headings=['Model', 'Stages', 'RMSE(a, recon(f))', 'PSNR(a, recon(f))'],
                          stages = True)

