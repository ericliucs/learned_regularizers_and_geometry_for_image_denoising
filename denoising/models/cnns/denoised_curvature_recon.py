from denoising.models.denoising_model import DenoisingModel
from tensorflow.keras.optimizers import Adam
from denoising.models.cnns.regularizers.regularizer_model import RegularizerModel
from denoising.models.cnns.vnet import VNetModel
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.initializers import Constant
import numpy as np
from denoising.util import MinMax
from typing import Dict
from denoising.models.cnns.regularizers.tv.tv import TVRegularizer


class DenoisedCurvatureRecon(tf.keras.Model):
    """Implements curvature reconstruction equation with curvature denoising model, F."""

    def __init__(self,
                 config: Dict,
                 F: tf.keras.Model = None,
                 D: tf.keras.Model = None,
                 **kwargs):
        """
        Initializes denoised curvature reconstruction process.

        Parameters
        ----------
        config: (Dict) - Dictionary containing configuration of model
        F: (tf.keras.Model) - Curvature denoising model
        """

        # Get settings
        super(DenoisedCurvatureRecon, self).__init__(**kwargs)
        self.config = config
        self.direct_denoising = config['direct_curvature_denoising']
        self.S = config['S']
        self.F = F
        self.D = D

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

        # Initialize curvature layer
        self.curvature = TVRegularizer({}).grad

    def call(self, inputs, training=None, mask=None):
        """Computes reconstruction"""

        # Grab input
        z = inputs
        x = z
        if self.direct_denoising:
            Fk = self.F(self.curvature(z))
        else:
            Fk = self.curvature(self.F(z))

        if self.D is not None:
            x = self.D(x)

        # Compute each stage of reconstruction
        for s in range(1, self.S + 1):
            x = x + self.T * (self.curvature(x) - Fk) + self.lmbda * (z - x)
        return x


class DenoisedCurvatureReconModel(DenoisingModel):
    """Model class for reconstructing with denoised curvature using a
    finite number of steps and trainable scaling weights."""

    def __init__(self,
                 config: Dict,
                 train: bool = True,
                 add_keys: bool = False,
                 multitraining: bool = False,
                 ):
        """Initializes denoised curvature reconstruction model.

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
        self._check_for_direct_or_indirect()
        self._check_for_type_of_denoising_model()
        super(DenoisedCurvatureReconModel, self).__init__(config=config, train=train, multitraining=multitraining)

    def _build(self):
        """Builds reconstruction model"""
        # Initialize curvature denoiser
        if not self.config['F_vnet']:
            self.F_model_class = RegularizerModel(deepcopy(self.config['F']), add_keys=False)
            self.F = self.F_model_class.model
        else:
            self.F_model_class = VNetModel(deepcopy(self.config['F']), add_keys=False)
            self.F = self.F_model_class.model

        if not self.config['F_train']:
            self.F.trainable = False

        if 'D' in self.config:
            self.D = VNetModel(deepcopy(self.config['D']), add_keys=False).model
            if not self.config['D_train']:
                self.D.trainable = False
        else:
            self.D = None
        return DenoisedCurvatureRecon(self.config, F=self.F, D=self.D)

    def _add_keys_to_config(self):
        """
        Keys added to config if self.add_keys is True
        """
        # Training settings
        self.config['training_type'] = 'standard'
        self.config['epochs'] = 10
        self.config['loss_function'] = 'mean_sum_squared_error_loss'
        self.config['num_training_images'] = 400
        self.config['patch_size'] = 64
        self.config['batch_size'] = 64
        self.config['grayscale'] = True
        self.config['test'] = 'BSDS68'
        self.config['test_task'] = 'denoising'
        self.config['train'] = 'BSDS400'
        self.config['train_task'] = 'denoising'
        self.config['sigma'] = 25

        # Reconstruction Process
        self.config['S'] = 12
        self.config['F_vnet'] = True
        self.config['F_train'] = False
        self.config['direct_curvature_denoising'] = False
        self.config['backprop_num'] = 1

        # Trained curvature denoiser
        self.config['F'] = {
            'epochs': 10,
            'loss_function': 'mean_sum_absolute_error_curvature_loss',
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
        }

    def _check_for_direct_or_indirect(self):
        """Checks if F setting is a direct or indirect denoiser"""
        if 'curvature' in self.config['F']['loss_function']:
            self.config['direct_curvature_denoising'] = False
        else:
            self.config['direct_curvature_denoising'] = True

    def _check_for_type_of_denoising_model(self):
        """Checks for type of denoising model, either full VNet or RegularizerModel"""
        if 'S' not in self.config['F']:
            self.config['F_vnet'] = False
        else:
            self.config['F_vnet'] = True

    def scheduler(self):
        """Defines scheduler for training"""
        if not self.config['F_train']:
            def lr_scheduler(epoch):
                initial_lr = 1e-3
                lr = initial_lr * ((1 / 2) ** (epoch // 2))
                return lr
            return lr_scheduler
        else:
            return self.F_model_class.scheduler()

    def optimizer(self):
        """Defines optimizer for training"""
        return Adam(lr=1e-3)

    def _train(self):
        """Trains the model based on specified training type"""
        if self.config['training_type'] == 'standard':
            self._standard_training(save_every=2, test_every=2)
        else:
            raise Exception(f'Training type {self.config["training_type"]} does not exist')
