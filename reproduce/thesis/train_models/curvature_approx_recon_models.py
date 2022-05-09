from models.cnns.approx_curvature_oracle_recon import ApproxCurvatureOracleReconModel


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


def curvature_approx_recon_models():
    models = []
    for S in range(1, 11):
        model = standard_approx_curvature_recon_settings()
        model['S'] = S
        models.append(model)

        model = standard_approx_curvature_recon_settings()
        model['S'] = S
        model['R'] = {}
        model['R']['name'] = 'TV'
        models.append(model)

    return models, ApproxCurvatureOracleReconModel