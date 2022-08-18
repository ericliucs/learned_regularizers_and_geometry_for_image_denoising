from denoising.util import getGPU
from denoising.models.cnns.vnet import VNetModel


def tnrd_config():
    """Returns config for small TNRD model"""
    return dict(epochs=10, loss_function='mean_sum_squared_error_loss', num_training_images=400, patch_size=50,
                batch_size=64, grayscale=True, test='BSDS68', test_task='denoising', train='BSDS400',
                train_task='denoising', sigma=25, training_type='standard', constant_dataterm_weight=False,
                scale_dataterm_weight=False, constant_regularizer=False, regularizer_weight=False,
                scale_regularizer_weight=False, use_prox=False, S=3, R={'name': 'TNRD',
                                                                        'filters': 8,
                                                                        'kernel_size': 3,
                                                                        'grayscale': True})


def train_test_tnrd():
    """Trains and tests small TNRD model"""

    print('Training small TNRD model')
    model = VNetModel(tnrd_config())

    print('Testing small TNRD model on BSDS68 with sigma=25')
    psnr_val = model.test()['PSNR']

    print(f'The PSNR value was {psnr_val}')


if __name__ == '__main__':
    getGPU()
    train_test_tnrd()
