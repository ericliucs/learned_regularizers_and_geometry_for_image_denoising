
from denoising.models.cnns.regularizers.regularizer_model import RegularizerModel
from denoising.models.cnns.vnet import VNetModel


def direct_curvature_denoisers():
    """Returns list of model types and configurations for direct curvature denoisers

    Returns
    -------
    List of tuples of size 2. First entry of tuple is model type. Second entry of tuple is configuration of the model.
    """
    return [
        (RegularizerModel, {
            'epochs': 10,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 64,
            'batch_size': 64,
            'grayscale': True,
            'test': 'BSDS68',
            'test_task': 'denoise_curvature',
            'train': 'BSDS400',
            'train_task': 'denoise_curvature',
            'sigma': 25,
            'training_type': 'standard',

            'R': {'name': 'TNRD',
                  'filters': 8,
                  'kernel_size': 3,
                  'grayscale': True}
        }),
        (RegularizerModel, {
            'epochs': 10,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 64,
            'batch_size': 64,
            'grayscale': True,
            'test': 'BSDS68',
            'test_task': 'denoise_curvature',
            'train': 'BSDS400',
            'train_task': 'denoise_curvature',
            'sigma': 25,
            'training_type': 'standard',

            'R': {'name': 'TNRD',
                  'filters': 24,
                  'kernel_size': 5,
                  'grayscale': True}
        }),
        (RegularizerModel, {
            'epochs': 10,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 64,
            'batch_size': 64,
            'grayscale': True,
            'test': 'BSDS68',
            'test_task': 'denoise_curvature',
            'train': 'BSDS400',
            'train_task': 'denoise_curvature',
            'sigma': 25,
            'training_type': 'standard',

            'R': {'name': 'TNRD',
                  'filters': 48,
                  'kernel_size': 7,
                  'grayscale': True}
         }),

        (VNetModel,
         {
             'epochs': 10,
             'loss_function': 'mean_sum_squared_error_loss',
             'num_training_images': 400,
             'patch_size': 64,
             'batch_size': 64,
             'grayscale': True,
             'test': 'BSDS68',
             'test_task': 'denoise_curvature',
             'train': 'BSDS400',
             'train_task': 'denoise_curvature',
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
         }),

        (VNetModel,
         {
             'epochs': 10,
             'loss_function': 'mean_sum_squared_error_loss',
             'num_training_images': 400,
             'patch_size': 64,
             'batch_size': 64,
             'grayscale': True,
             'test': 'BSDS68',
             'test_task': 'denoise_curvature',
             'train': 'BSDS400',
             'train_task': 'denoise_curvature',
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
         }),

        (VNetModel,
        {
            'epochs': 10,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 64,
            'batch_size': 64,
            'grayscale': True,
            'test': 'BSDS68',
            'test_task': 'denoise_curvature',
            'train': 'BSDS400',
            'train_task': 'denoise_curvature',
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
        }),
    ]


def curvature_denoisers():
    """Returns list of model types and configurations for all curvature denoisers

      Returns
      -------
      List of tuples of size 2. First entry of tuple is model type. Second entry of tuple
          is configuration of the model.
    """
    list_of_model_types_and_configs = []
    for model_type, config in direct_curvature_denoisers():
        config['train_task'] = 'denoising'
        config['test_task'] = 'denoising'
        config['loss_function'] = 'mean_sum_squared_error_curvature_loss'
        list_of_model_types_and_configs.append((model_type, config))
    list_of_model_types_and_configs += direct_curvature_denoisers()
    return list_of_model_types_and_configs

