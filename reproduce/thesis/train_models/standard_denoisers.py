from models.cnns.vnet import VNetModel


def standard_denoiser_configs():
    return [
        # TNRD
        {
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
                  'filters': 8,
                  'kernel_size': 3,
                  'grayscale': True}
        },

               {
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
                         'filters': 24,
                         'kernel_size': 5,
                         'grayscale': True}
               },

               {
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
               },

        # DnCNN
        # {
        #     'epochs': 30,
        #     'loss_function': 'mean_sum_squared_error_loss',
        #     'num_training_images': 400,
        #     'patch_size': 64,
        #     'batch_size': 64,
        #     'grayscale': True,
        #     'test': 'BSDS68',
        #     'test_task': 'denoising',
        #     'train': 'BSDS400',
        #     'train_task': 'denoising',
        #     'sigma': 25,
        #     'training_type': 'standard',
        #
        #     'constant_dataterm_weight': False,
        #     'constant_regularizer': False,
        #     'descent_weight': False,
        #     'scale_descent_weight': False,
        #     'scale_dataterm_weight': False,
        #     'use_prox': False,
        #     'S': 1,
        #
        #     'R': {'name': 'DnCNN',
        #           'filters': 64,
        #           'depth': 17,
        #           'grayscale': True}
        # },

        # {
        #     'epochs': 2,
        #     'loss_function': 'mean_sum_squared_error_loss',
        #     'num_training_images': 400,
        #     'patch_size': 50,
        #     'batch_size': 64,
        #     'grayscale': True,
        #     'test': 'BSDS68',
        #     'test_task': 'denoising',
        #     'train': 'BSDS400',
        #     'train_task': 'denoising',
        #     'sigma': 25,
        #     'training_type': 'layer_wise',
        #
        #     'constant_dataterm_weight': True,
        #     'constant_regularizer': True,
        #     'descent_weight': True,
        #     'scale_descent_weight': True,
        #     'scale_dataterm_weight': True,
        #     'use_prox': True,
        #     'S': 10,
        #
        #     'R': {'name': 'TDV',
        #           'filters': 32,
        #           'num_scales': 3,
        #           'multiplier': 1,
        #           'num_mb': 3,
        #             'grayscale': True}
        # },
    ], VNetModel