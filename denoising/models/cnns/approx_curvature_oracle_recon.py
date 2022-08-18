from denoising.models.denoising_model import DenoisingModel
from tensorflow.keras.optimizers import Adam
from denoising.models.cnns.regularizers.regularizer_model import RegularizerModel, retrieve_regularizer
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.initializers import Constant
import numpy as np
from denoising.util import MinMax
from typing import Dict


class ApproxCurvatureOracleRecon(tf.keras.Model):
    """Implements finite curvature reconstruction with approximation of curvature and learnable scaling weights"""

    def __init__(self,
                 config: Dict,
                 R: tf.keras.Model,
                 **kwargs):
        """
        Initializes variational network

        Parameters
        ----------
        config: (Dict) - Dictionary containing configuration of model
        R: ML Model that approximates the curvature operator
        """

        # Get settings
        super(ApproxCurvatureOracleRecon, self).__init__(name='Recon', **kwargs)
        self.config = config
        self.S = config['S']
        self.R = R

        # Initialize scaling weights
        self.T = self.add_weight(name='T',
                                 shape=[1],
                                 initializer=Constant(np.asarray(0.001)),
                                 constraint=MinMax(min_value=0, max_value=1000),
                                 trainable=True)

        self.lmbda = self.add_weight(name='lmbda',
                                     shape=[1],
                                     initializer=Constant(np.asarray(0.01)),
                                     constraint=MinMax(min_value=0, max_value=1000),
                                     trainable=True)

    def call(self, inputs, training=None, mask=None):
        """Computes oracle reconstruction process for S stages"""

        # Noisy input
        z = inputs[0]

        # Clean input
        a = inputs[1]

        # Compute each stage of reconstruction
        x = z
        for s in range(1, self.S+1):
            x = x + self.T*(self.R(x) - self.R(a)) + self.lmbda*(z-x)
        return x


class ApproxCurvatureOracleReconModel(DenoisingModel):
    """Approximate Curvature Oracle Reconstruction model class. Mimics a finite curvature oracle reconstruction
    process but a learned approximate curvature operator is used."""
    def __init__(self,
                 config: Dict,
                 train: bool = True,
                 add_keys: bool = False,
                 multitraining: bool = False,
                 ):
        """Initializes the model

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
        super(ApproxCurvatureOracleReconModel, self).__init__(config=config, train=train, multitraining=multitraining)

    def _build(self):
        """Initializes trained regularizer model and returns full reconstruction model """

        # Get ml approximation of curvature
        if 'name' in self.config['R']:
            self.R = retrieve_regularizer(self.config['R'])
        else:
            self.R = RegularizerModel(deepcopy(self.config['R']), add_keys=False).model
        self.R.trainable = False

        # Initialize full oracle reconstruction model
        model = ApproxCurvatureOracleRecon(self.config, self.R)
        return model

    def _add_keys_to_config(self):
        # Training
        self.config['training_type'] = 'standard'
        self.config['epochs'] = 10
        self.config['loss_function'] = 'mean_sum_squared_error_loss'
        self.config['num_training_images'] = 400
        self.config['patch_size'] = 64
        self.config['batch_size'] = 64
        self.config['grayscale'] = True
        self.config['test'] = 'BSDS68'
        self.config['test_task'] = 'oracle_recon'
        self.config['train'] = 'BSDS400'
        self.config['train_task'] = 'oracle_recon'
        self.config['sigma'] = 25

        # VNet Process
        self.config['S'] = 10

        # # Regularizer
        self.config['R'] = {
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
                                  'grayscale': True}},

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
