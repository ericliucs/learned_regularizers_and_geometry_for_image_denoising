from typing import List
from models.denoising_model import DenoisingModel
import os
from training.losses import retrieve_loss_function
from tensorflow.keras.models import load_model
from copy import deepcopy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.initializers import Constant
import numpy as np
from util import MinMax
from typing import Dict
from models.cnns.regularizers.retrieve_regularizer import retrieve_regularizer
from util import L2DenoiseDataterm


class VNet(tf.keras.Model):
    """Implementation of denoising variational network.
    Code based on https://github.com/VLOGroup/tdv/blob/master/model.py
    Adds capability for different regularizers and scaling weights"""

    def get_config(self):
        pass

    def __init__(self,
                 config: Dict,
                 **kwargs):
        """
        Initializes variational network

        Parameters
        ----------
        config: (Dict) - Dictionary containing configuration of model.
        """
        super(VNet, self).__init__(name='VNet', **kwargs)
        self.config = config
        self.S = config['S']
        self.constant_regularizer = config['constant_regularizer']
        self.use_prox = config['use_prox']
        self.constant_dataterm_weight = config['constant_dataterm_weight']
        self.scale_dataterm_weight = config['scale_dataterm_weight']
        self.regularizer_weight = config['regularizer_weight']
        self.scale_regularizer_weight = config['scale_regularizer_weight']

        # Initialize scaling weights and data term (as long as we aren't initalizing DnCNN)
        if not self.config['R']['name'] == 'DnCNN':
            if self.regularizer_weight:
                self.T = self.add_weight(name='T',
                                         shape=[1],
                                         initializer=Constant(np.asarray(0.001)),
                                         constraint=MinMax(min_value=0, max_value=1000),
                                         trainable=True)

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

        # Initialize regularizer parameters
        if self.constant_regularizer:
            self.R = retrieve_regularizer(config['R'])
        else:
            self.R = [retrieve_regularizer(config['R']) for _ in range(self.S)]

    def call(self, inputs, training=None, mask=None):
        """Compute variational network"""

        # Get inputs, save initial input
        x = inputs

        # Apply only residual if DnCNN
        if self.config['R']['name'] == 'DnCNN':
            if isinstance(self.R, list):
                regularizer = self.R[0].grad
            else:
                regularizer = self.R.grad
            x = x - regularizer(x)
        # else, compute variational network
        else:
            # store initial image
            z = x

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

            # Grab regularizer if constant
            if self.constant_regularizer:
                regularizer_grad = self.R.grad

            # Compute each stage of variational network
            for s in range(1, self.S+1):

                # Grab regularizer if not constant
                if not self.constant_regularizer:
                    regularizer_grad = self.R[s - 1].grad

                grad_r = regularizer_grad(x)

                # Apply descent weight
                if self.regularizer_weight:
                    grad_r = tau*grad_r

                # Retrieve data term weight if not constant, and then scale it
                if not self.constant_dataterm_weight:
                    lmbda = self.lmbda[s-1]
                    if self.scale_dataterm_weight:
                        lmbda = lmbda / self.S

                # If specified use proximal descent step, otherwise use standard gradient descent step
                if self.use_prox:
                    x = self.D.prox(x - grad_r, z, lmbda)
                else:
                    x = x - grad_r - lmbda * self.D.grad(x, z)
        return x


class VNetModel(DenoisingModel):
    """Variational Network Model class. Able to implement and train
    finite TV, TNRD, DnCNN, and TDV denoising models."""
    def __init__(self,
                 config: Dict,
                 train: bool=True,
                 add_keys: bool=True,
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
        self._check_keys(self.config, self._vnet_keys() + self._training_keys(), exceptions=['len_test_1'])
        super(VNetModel, self).__init__(config=config, train=train, multitraining=multitraining)

    def _build(self):
        """Initializes regularizer and returns model """
        model = VNet(deepcopy(self.config))
        self.vnet = model
        self.R = model.R
        return model

    def _add_keys_to_config(self):
        """
        Keys added to config if self.add_keys is True
        """
        # Training
        self.config['training_type'] = 'layer_wise'
        self.config['epochs'] = 2
        self.config['loss_function'] = 'mean_sum_squared_error_loss'
        self.config['num_training_images'] = 400
        self.config['patch_size'] = 40
        self.config['batch_size'] = 64
        self.config['grayscale'] = True
        self.config['test'] = 'BSDS68'
        self.config['test_task'] = 'denoising'
        self.config['train'] = 'BSDS400'
        self.config['train_task'] = 'denoising'
        self.config['sigma'] = 25

        # VNet Process
        self.config['constant_dataterm_weight'] = False
        self.config['scale_dataterm_weight'] = False
        self.config['constant_regularizer'] = False
        self.config['regularizer_weight'] = False
        self.config['scale_regularizer_weight'] = False
        self.config['use_prox'] = False
        self.config['S'] = 10

        # # Regularizer
        self.config['R'] = {'name': 'TNRD',
                            'filters': 48,
                            'kernel_size': 7,
                            'grayscale': True}

    @staticmethod
    def _vnet_keys() -> List:
        """Returns keys necessary for variational network configuration"""
        return ['S', 'constant_dataterm_weight', 'constant_regularizer', 'scale_dataterm_weight',
                'regularizer_weight', 'scale_regularizer_weight', 'use_prox', 'R']

    def scheduler(self):
        """Defines scheduler for training depending on model"""
        if self.config['R']['name'] == 'TDV':
            if self.config['training_type'] == 'standard':
                def lr_scheduler(epoch):
                    initial_lr = 4e-4
                    lr = initial_lr * ((1 / 2) ** (epoch // 2))
                    return lr
            elif self.config['training_type'] == 'layer_wise':
                def lr_scheduler(epoch):
                    initial_lr = 4e-4
                    lr = initial_lr * ((1 / 2) ** (epoch // 1))
                    return lr
        elif self.config['R']['name'] == 'DnCNN':
            def lr_scheduler(epoch):
                initial_lr = 1e-3
                if epoch <= 6:
                    lr = initial_lr
                elif epoch <= 12:
                    lr = initial_lr / 10
                else:
                    lr = initial_lr / 20
                return lr
        else:
            def lr_scheduler(epoch):
                initial_lr = 1e-3
                lr = initial_lr * ((1 / 2) ** (epoch // 2))
                return lr
        return lr_scheduler

    def optimizer(self):
        """Defines optimizer for training"""
        return Adam(lr=1e-3)

    def _clear_all_model_data(self):
        """
        Clears all current model data from tensorflow graph
        """
        del self.model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    def _train_layer_wise(self):
        """Layer wise training for model.
        Continually trains model and after self.epochs,
        adds a single stage to the model. Stops training once the model has been trained with S stages."""

        if self.config['S'] > 1:

            def load_model_from_file(file: str):
                """Clears the keras session and loads model from file if file exists. Otherwise, returns None.

                Parameters
                ----------
                file: (str) - File which contains keras model

                Returns
                -------
                model: Keras model if file exists. None otherwise.
                """
                if os.path.exists(file):
                    model = self._build()
                    model = self._model(model)
                    model.compile(optimizer=self.optimizer(), loss=retrieve_loss_function(self.config))
                    loaded_model = load_model(file, custom_objects={
                        self.config['loss_function']: retrieve_loss_function(self.config)})
                    model.set_weights(loaded_model.get_weights())
                    return model
                else:
                    return None

            # Attempt to load init if it exists
            self._clear_all_model_data()
            self.model = load_model_from_file(os.path.join(self.data_dir, 'init'))

            # Otherwise grab init from model with S-1 stages and train if necessary
            if self.model is None:

                # Ensure previous model is trained
                tf.keras.backend.clear_session()
                self.config['S'] = self.config['S'] - 1
                self._clear_all_model_data()
                prev_model = type(self)(self.config, add_keys=False)
                self.config['S'] = self.config['S'] + 1

                # Reload model
                self.model = self._load_model(self.epochs)

                # Replace weights
                vnet = self.model.get_layer('VNet')
                prev_vnet = prev_model.model.get_layer('VNet')
                vnet.R.set_weights(prev_vnet.R.get_weights())
                vnet.T.assign(prev_vnet.T)
                vnet.lmbda.assign(prev_vnet.lmbda)

                # Save loaded model
                self.model.save(os.path.join(self.data_dir, 'init'))
                del prev_model
                self._clear_all_model_data()
                self.model = load_model_from_file(os.path.join(self.data_dir, 'init'))

        print(f'Training {self.config["S"]} Stage Model')
        self._standard_training(save_every=2, test_every=2)

    def _train(self):
        """Trains the model based on specified training type"""
        if self.config['training_type'] == 'standard':
            self._standard_training(save_every=2, test_every=2)
        elif self.config['training_type'] == 'layer_wise':
            self._train_layer_wise()
        else:
            raise Exception(f'Training type {self.config["training_type"]} does not exit')
