from models.cnns.regularizers.regularizer_model import RegularizerModel


def standard_approx_curvature_training_settings():
    return {
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
            'training_type': 'standard'
    }


def partial_curvature_regularizers():
    return [

        {'name': 'TNRD',
         'filters': 1,
         'kernel_size': 3,
         'grayscale': True},

        {'name': 'KTNRD',
         'filters': 1,
         'kernel_size': 3,
         'grayscale': True},

        {'name': 'KTNRD',
         'filters': 8,
         'kernel_size': 3,
         'grayscale': True},

        {'name': 'KTNRD',
         'filters': 24,
         'kernel_size': 5,
         'grayscale': True},

        {'name': 'KTNRD',
         'filters': 48,
         'kernel_size': 7,
         'grayscale': True},
    ]


def full_curvature_regularizers():
    return [
        # TNRD
            {'name': 'TNRD',
                  'filters': 8,
                  'kernel_size': 3,
                  'grayscale': True},

        {'name': 'TNRD',
         'filters': 24,
         'kernel_size': 5,
         'grayscale': True},

        {'name': 'TNRD',
         'filters': 48,
         'kernel_size': 7,
         'grayscale': True},

        # DnCNN
        # {'name': 'DnCNN',
        #           'filters': 64,
        #           'depth': 17,
        #           'grayscale': True},

        # # TDV
        # {'name': 'TDV',
        #           'filters': 32,
        #           'num_scales': 3,
        #           'multiplier': 1,
        #           'num_mb': 3,
        #           'grayscale': True},
    ]


def curvature_regularizers():
    return partial_curvature_regularizers() + full_curvature_regularizers()


def full_curvature_approx_models():
    models = []
    for regularizer in full_curvature_regularizers():
        config = standard_approx_curvature_training_settings()
        if regularizer['name'] == 'TDV':
            config['patch_size'] = 50
        config['R'] = regularizer
        models.append(config)
    return models, RegularizerModel

def partial_curvature_approx_models():
    models = []
    for regularizer in partial_curvature_regularizers():
        config = standard_approx_curvature_training_settings()
        if regularizer['name'] == 'TDV':
            config['patch_size'] = 50
        config['R'] = regularizer
        models.append(config)
    return models, RegularizerModel


def curvature_approx_models():
    models = []
    for regularizer in curvature_regularizers():
        config = standard_approx_curvature_training_settings()
        if regularizer['name'] == 'TDV':
            config['patch_size'] = 50
        config['R'] = regularizer
        models.append(config)
    return models, RegularizerModel

