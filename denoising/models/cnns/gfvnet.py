from denoising.models.denoising_model import DenoisingModel
from copy import deepcopy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from denoising.util import MinMax
from denoising.models.cnns.regularizers.retrieve_regularizer import retrieve_regularizer
from denoising.util import L2DenoiseDataterm
from typing import Dict, List
import numpy as np
from denoising.models.cnns.vnet import VNet


class GFVNet(tf.keras.Model):
    """Implementation of GF denoising variational network.
    Code based on https://github.com/VLOGroup/tdv/blob/master/model.py
    Adds capability for different learned geometry and regularizer terms and scaling weights"""

    def __init__(self,
                 config: Dict,
                 **kwargs):
        """
        Initializes variational network

        Parameters
        ----------
        config: (Dict) - Dictionary containing configuration of model.
        """
        super(GFVNet, self).__init__(**kwargs)

        # Set variables
        self.config = config
        self.S = config['S']
        self.constant_geometry = config['constant_geometry']
        self.constant_denoiser = config['constant_denoiser']
        self.use_prox = config['use_prox']
        self.constant_dataterm_weight = config['constant_dataterm_weight']
        self.scale_dataterm_weight = config['scale_dataterm_weight']
        self.regularizer_weight = config['regularizer_weight']
        self.constant_regularizer_weight = config['constant_regularizer_weight']
        self.scale_regularizer_weight = config['scale_regularizer_weight']
        self.direct_geometry_denoising = config['direct_geometry_denoising']
        self.res_geometry_denoising = config['res_geometry_denoising']
        self.steps_per_geometry = config['steps_per_geometry']

        # Initialize scaling weights and data term
        if self.regularizer_weight:
            if self.constant_regularizer_weight:
                self.T = self.add_weight(name='T',
                                         shape=[1],
                                         initializer=Constant(np.asarray(0.001)),
                                         constraint=MinMax(min_value=0, max_value=1000),
                                         trainable=True)
            else:
                self.T = [self.add_weight(name=f'T_{i}',
                                          shape=[1],
                                          initializer=Constant(np.asarray(0.001)),
                                          constraint=MinMax(min_value=0, max_value=1000),
                                          trainable=True) for i in range(self.S)]

        if self.constant_dataterm_weight:
            self.lmbda = self.add_weight(name='lmbda',
                                         shape=[1],
                                         initializer=Constant(np.asarray(0.01)),
                                         constraint=MinMax(min_value=0, max_value=1000),
                                         trainable=True)
        else:
            self.lmbda = [self.add_weight(name=f'lmbda_{i}',
                                          shape=[1],
                                          initializer=Constant(np.asarray(0.01)),
                                          constraint=MinMax(min_value=0, max_value=1000),
                                          trainable=True)
                          for i in range(self.S)]

        # Get dataterm
        self.D = L2DenoiseDataterm()

        # Check if denoiser is a full variational network
        if 'R' in config['F']:
            self.F_vnet = True
            F_fnc = VNet
        else:
            self.F_vnet = False
            F_fnc = retrieve_regularizer

        # Grab geometry terms
        if self.constant_geometry:
            self.G = retrieve_regularizer(config['G'])
        else:
            self.G = [retrieve_regularizer(config['G']) for _ in range(self.S)]

        # Grab denoiser terms
        if self.constant_denoiser:
            self.F = F_fnc(config['F'])
        else:
            self.F = [F_fnc(config['F']) for _ in range(self.S)]

    def compute_operator(self, x, operator, vnet: bool = False):
        """Computes denoising operator on input x"""

        # Check if we have full vnet denoising operator
        if vnet:
            fnc = operator
        else:
            fnc = operator.grad

        # Apply residual denoising of geometry if specified
        if self.res_geometry_denoising:
            return x - fnc(x)
        else:
            return fnc(x)

    def call(self, inputs, training=None, mask=None):
        """Compute GF variational network"""

        # Get inputs
        z = inputs
        x = z

        # Grab descent weight and scale it by S if constant
        if self.regularizer_weight:
            tau = self.T
            if self.scale_regularizer_weight:
                tau = tau / self.S

        # Grab dataterm weight and scale by S if constant
        if self.constant_dataterm_weight:
            lmbda = self.lmbda
            if self.scale_dataterm_weight:
                lmbda = lmbda / self.S

        # Grab geometry if constant
        if self.constant_geometry:
            geometry = self.G

        # Grab denoiser or geometry if constant; if both are constant, then compute denoising
        if self.constant_denoiser:
            denoiser = self.F
            if self.constant_geometry:
                if self.direct_geometry_denoising:
                    fgz = self.compute_operator(geometry(z), denoiser, self.F_vnet)
                else:
                    fgz = geometry(self.compute_operator(z, denoiser, self.F_vnet))

        # Grab regularizer weight if specified
        if self.regularizer_weight:
            if self.constant_regularizer_weight:
                tau = self.T

        # Compute each stage of GF variational network
        for s in range(1, self.S + 1):

            # Grab geometry and/or denoiser if they are not constant
            if not self.constant_geometry:
                geometry = self.G[s - 1]
                if not self.constant_denoiser:
                    denoiser = self.F[s - 1]
                if self.direct_geometry_denoising:
                    fgz = self.compute_operator(geometry(z), denoiser, self.F_vnet)
                else:
                    fgz = geometry(self.compute_operator(z, denoiser, self.F_vnet))

            # Grab regularizer weight if not constant
            if self.regularizer_weight:
                if not self.constant_regularizer_weight:
                    tau = self.T[s - 1]

            # Retrieve data term weight if not constant, and then scale it
            if not self.constant_dataterm_weight:
                lmbda = self.lmbda[s - 1]
                if self.scale_dataterm_weight:
                    lmbda = lmbda / self.S

            # Apply self.steps_per_geometry reconstruction iterations if specified
            for _ in range(self.steps_per_geometry):

                gx = geometry(x)
                grad_r = gx - fgz

                if self.regularizer_weight:
                    grad_r = tau * grad_r

                # If specified use recon equation form, prox, or regular descent
                if self.config['use_recon']:
                    x = x + grad_r + lmbda * (z - x)
                else:
                    if self.config['use_prox']:
                        x = self.D.prox(x - grad_r, z, lmbda)
                    else:
                        x = x - grad_r - lmbda * self.D.grad(x, z)
        return x


class GFVNetModel(DenoisingModel):
    """GF Variational Network Model class. Able to implement and train
    GF models with TV, TNRD, DnCNN, and TDV used as the regularizer and/or denoiser terms."""

    def __init__(self,
                 config: Dict,
                 train: bool = True,
                 add_keys: bool = False,
                 multitraining: bool = False,
                 ):
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
        self._check_keys(self.config, self._training_keys() + self._vnet_keys(), exceptions=['len_test_1'])
        super(GFVNetModel, self).__init__(config=config, train=train, multitraining=multitraining)

    def _build(self):
        """Initializes regularizer and returns model"""
        model = GFVNet(deepcopy(self.config))
        self.vnet = model
        self.G = model.G
        self.F = model.F
        return model

    def _check_type_of_denoising(self):
        """Checks if we are curvature denoising"""
        if 'direct_curvature_denoising' in self.config:
            if self.config['direct_curvature_denoising']:
                self.config['loss_function'] = 'mean_sum_squared_error_curvature_loss'
                self.config['train_task'] = 'denoising'
                self.config['test_task'] = 'denoising'
            else:
                self.config['loss_function'] = 'mean_sum_squared_error_loss'
                self.config['train_task'] = 'denoise_curvature'
                self.config['test_task'] = 'denoise_curvature'
            del self.config['direct_curvature_denoising']

    def _add_keys_to_config(self):
        """
        Keys added to config if self.add_keys is True, example of regular GFTNRD model
        """
        return dict(training_type='standard', epochs=10, loss_function='mean_sum_squared_error_loss',
                    num_training_images=400, patch_size=50, batch_size=64, grayscale=True, test='BSDS68',
                    test_task='denoising', train='BSDS400', train_task='denoising', sigma=25,
                    constant_dataterm_weight=False, scale_dataterm_weight=False, regularizer_weight=False,
                    scale_regularizer_weight=False, constant_regularizer_weight=False,
                    constant_geometry=False,
                    constant_denoiser=False, use_prox=False, S=5, direct_geometry_denoising=True,
                    res_geometry_denoising=False,
                    steps_per_geometry=1, use_recon=False, G={'name': 'TNRD',
                                                              'filters': 48,
                                                              'kernel_size': 7,
                                                              'grayscale': True}, F={'name': 'TNRD',
                                                                                     'filters': 48,
                                                                                     'kernel_size': 7,
                                                                                     'grayscale': True})

    @staticmethod
    def _vnet_keys() -> List:
        """Returns keys necessary for GF variational network configuration"""
        return ['S', 'constant_dataterm_weight', 'constant_geometry', 'scale_dataterm_weight',
                'regularizer_weight', 'scale_regularizer_weight', 'use_prox', 'F', 'G']

    def scheduler(self):
        """Defines scheduler for training depending on model"""
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
