from models.cnns.vnet import VNetModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def getGPU():
    """
    Grabs GPU. Sometimes Tensorflow attempts to use CPU when this is not called on my machine.
    From: https://www.tensorflow.org/guide/gpu
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def model_configs():
    return [
        {
            'epochs': 10,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 40,
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
        },

        {
            'epochs': 20,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 40,
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
            'S': 1,

            'R': {'name': 'DnCNN',
                  'filters': 64,
                  'depth': 17,
                  'grayscale': True}
        },

        {
            'epochs': 2,
            'loss_function': 'mean_sum_squared_error_loss',
            'num_training_images': 400,
            'patch_size': 40,
            'batch_size': 64,
            'grayscale': True,
            'test': 'BSDS68',
            'test_task': 'denoising',
            'train': 'BSDS400',
            'train_task': 'denoising',
            'sigma': 25,
            'training_type': 'layer_wise',

            'constant_dataterm_weight': True,
            'scale_dataterm_weight': True,
            'constant_regularizer': True,
            'regularizer_weight': True,
            'scale_regularizer_weight': True,
            'use_prox': True,
            'S': 10,

            'R': {'name': 'TDV',
                  'filters': 32,
                  'num_scales': 3,
                  'multiplier': 1,
                  'num_mb': 3,
                    'grayscale': True}
        },
    ]


if __name__ == '__main__':
    getGPU()
    for config in model_configs():
        model = VNetModel(config, add_keys=False)
        del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
