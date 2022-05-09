from models.denoising_model import DenoisingModel
from typing import Dict
from tensorflow.keras.optimizers import Adam
from models.cnns.regularizers.retrieve_regularizer import retrieve_regularizer
from copy import deepcopy


class RegularizerModel(DenoisingModel):
    """Model class for single regularizer gradient"""
    def __init__(self,
                 config: Dict,
                 train: bool=True,
                 multitraining: bool=False,
                 add_keys: bool=True):
        """Initializes the variational network model

        Parameters
        ----------
        config: (Dict) - Configuration settings for model
        train: (bool) - If True, model will be trained if it has not already been
        add_keys: (bool) - If True, adds keys specified by _add_keys_to_config method to configuration.
        multitraining: (bool) - Specify as true if training the models across multiple computers which are
            using the same memory space to store the results.
        """
        self.config = config
        if add_keys:
            self._add_keys_to_config()
        super(RegularizerModel, self).__init__(config=config, train=train, multitraining=multitraining)

    def _build(self):
        """Initializes regularizer and returns model """
        self.R = retrieve_regularizer(deepcopy(self.config['R']))
        return self.R

    def _add_keys_to_config(self):
        """
        Keys added to config if self.add_keys is True
        """
        self.config['epochs'] = 10
        self.config['loss_function'] = 'mean_sum_squared_error_loss'
        self.config['num_training_images'] = 400
        self.config['patch_size'] = 64
        self.config['batch_size'] = 64
        self.config['grayscale'] = True
        self.config['test'] = 'BSDS68'
        self.config['test_task'] = 'approx_curvature'
        self.config['train'] = 'BSDS400'
        self.config['train_task'] = 'approx_curvature'
        self.config['sigma'] = 25
        self.config['training_type'] = 'standard'

        self.config['R'] = {'name': 'TNRD',
                            'filters': 8,
                            'kernel_size': 3,
                            'grayscale': True}

    def scheduler(self):
        """Defines scheduler for training"""
        def lr_scheduler(epoch):
            initial_lr = 1e-3
            lr = initial_lr * ((1 / 2) ** (epoch // 2))
            return lr
        return lr_scheduler

    def optimizer(self):
        """Defines optimizer for training"""
        return Adam(lr=1e-3)

    def _train(self):
        """Trains the model based on specified training type"""
        if self.config['training_type'] == 'standard':
            self._standard_training(save_every=2, test_every=2)
        else:
            raise Exception(f'Training type {self.config["training_type"]} does not exit')

