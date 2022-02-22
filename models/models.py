from models.model import CNNModel
from models.vnets.tdv.tdv_vnet import TDVNet
from models.vnets.tnrd.tnrd_vnet import TNRDVNet
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from util import get_num_channels

class TNRD(CNNModel):
    """Implemented TNRD Model"""

    def _check_config_for_model_keys(self):
        pass

    def _add_keys_to_config(self):
        # Training
        self.config['epochs'] = 100
        self.config['training_function'] = 'standard'
        self.config['loss_function'] = 'sum_squared_error_loss'
        self.config['R'] = 400
        self.config['patch_size'] = 64
        self.config['batch_size'] = 64
        self.config['grayscale'] = True

        self.config['test'] = 'BSDS68'
        self.config['test_task'] = 'denoising'
        self.config['train'] = 'BSDS400'
        self.config['train_task'] = 'denoising'
        self.config['sigma'] = 25

        # Model
        self.config['stages'] = 3
        self.config['kernel_size'] = 3
        self.config['filters'] = 8

    def scheduler(self):
        def lr_scheduler(epoch):
            initial_lr = 0.001
            if epoch <= 2:
                lr = initial_lr
            elif epoch <= 6:
                lr = initial_lr / 10
            elif epoch <= 8:
                lr = initial_lr / 20
            else:
                lr = initial_lr / 30
            return lr
        return ExponentialDecay(initial_learning_rate=0.001, decay_steps=500, decay_rate=0.96)

    def optimizer(self):
        return Adam(lr=0.001)

    def _model(self):
        """Defines Keras model of full TNRD denoising regularization process

        Returns
        -------
        model: training.Model
            A Keras Functional model object that encompasses the TDV regularization process.
        """
        if self.config['grayscale']:
            num_channels = 1
        else:
            num_channels = 3
        noisy = Input(shape=(None, None, num_channels), name='noisy')
        x = TNRDVNet(S = self.config['stages'],
                     num_channels=num_channels,
                     features = self.config['filters'],
                     kernel_size = self.config['kernel_size'])(noisy)
        model = Model(inputs=noisy, outputs=x)
        return model

    def visualize_parameters(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError


from tensorflow.python.keras import backend as K
class TotalDeepVariation(CNNModel):
    def _check_config_for_model_keys(self):
        pass

    def _add_keys_to_config(self):
        # Training
        self.config['epochs'] = 100
        self.config['training_function'] = 'standard'
        self.config['loss_function'] = 'sum_squared_error_loss'
        self.config['R'] = 400
        self.config['patch_size'] = 93
        self.config['batch_size'] = 32
        self.config['grayscale'] = True

        self.config['test'] = 'BSDS68'
        self.config['test_task'] = 'denoising'
        self.config['train'] = 'BSDS400'
        self.config['train_task'] = 'denoising'
        self.config['sigma'] = 25
        self.config['checkpoint'] = True

        # Model
        self.config['S'] = 10
        self.config['filters'] = 32
        self.config['num_scales'] = 3
        self.config['num_mb'] = 3
        self.config['multiplier'] = 1
        self.config['use_prox'] = False

    def scheduler(self):
        def lr_scheduler(epoch):
            initial_lr = 4e-4
            lr = initial_lr*((1/2)**(epoch // 25))
            return lr
        return lr_scheduler

    def optimizer(self):
        return Adam(lr=4*(10**(-4)))

    def _model(self):
        """Defines Keras model of full TDV denoising regularization process

        Returns
        -------
        model: training.Model
            A Keras Functional model object that encompasses the TDV regularization process.
        """
        noisy = Input(shape=(None, None, get_num_channels(self.config)), name='noisy')
        x = TDVNet(self.config)(noisy)
        model = Model(inputs=noisy, outputs=x)
        return model

    def visualize_parameters(self):
        pass

    def test(self):
        pass