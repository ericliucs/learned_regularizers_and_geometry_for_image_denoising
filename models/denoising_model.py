from abc import ABC, abstractmethod
from typing import Dict, List
import os
import yaml
import glob
import re
from tensorflow.keras.models import load_model
from tensorflow.python.keras.engine import training
from generator.datagenerator import DataGenerator
import numpy as np
from training.losses import retrieve_loss_function
from tensorflow.keras import Input, Model
from util import get_num_channels, psnr, rmse
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from training.callbacks import PSNRTest


class DenoisingModel(ABC):
    """Defines abstract class for denoising models"""

    def __init__(self,
                 config: Dict,
                 train: bool = True,
                 multitraining: bool = False):
        """Initializes the denoising model

        Parameters
        ----------
        config: (Dict) - Configuration settings for model
        train: (bool) - If True, model will be trained if it has not already been
        multitraining: (bool) - Specify as true if training the models across multiple computers which are
            using the same memory space to store the results.
        """

        self.config = config

        # Find or create model directory
        self.models_dir = os.path.join(os.getcwd(), 'models', 'saved_models')
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)
        self.model_dir = self._find_model_dir()
        self.data_dir = os.path.join(self.model_dir, 'data')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        print(self.data_dir)

        # Create processing file to dictate whether model is currently being trained on seperate computer
        self.processing_file = os.path.join(self.data_dir, 'training.txt')
        self._make_processing_file()

        # If multitraining, warn other computers that this current model is being trained
        start_training = True
        if multitraining:
            if self._being_processed():
                start_training = False
            else:
                self._processing()

        # Start training
        if start_training:
            self.epochs = self._find_last_checkpoint()
            # Load model and train if not trained
            if self.epochs >= self.config['epochs']:
                self.model = self._load_model(self.config['epochs'])
            else:
                self.model = self._load_model(self.epochs)
                if train:
                    self._train()
                    self.epochs = self.config['epochs']
            self._not_processing()

    def _make_processing_file(self):
        """
        Makes processing txt file.
        """
        if not os.path.isfile(self.processing_file):
            self._not_processing()

    def _being_processed(self):
        """Reads processing file. Returns True if model is being trained (i.e. processing file contains a 1).

        Returns
        -------
        (bool): Returns true if model is already being trained by a different computer.

        """
        with open(self.processing_file, 'r') as txt_file:
            value = int(txt_file.readline()[0])
            return value == 1

    def _processing(self):
        """
        Writes 1 to processing file to symbolize that model is being trained by seperate computer.
        """
        with open(self.processing_file, 'w') as txt_file:
            txt_file.write('1')

    def _not_processing(self):
        """
        Writes 0 to processing file to symbolize that model is not being trained by seperate computer.
        """
        with open(self.processing_file, 'w') as txt_file:
            txt_file.write('0')

    def _model(self, keras_model):
        """Defines CNN Keras model

        Returns
        -------
        model: training.Model
            A Keras Functional model
        """

        # Add functionality for two inputs for oracle reconstruction
        noisy = Input(shape=(None, None, get_num_channels(self.config)), name='noisy')
        if self.config['train_task'] == 'oracle_recon':
            clean = Input(shape=(None, None, get_num_channels(self.config)), name='clean')
            input = [noisy, clean]
        else:
            input = noisy

        x = keras_model(input)
        model = Model(inputs=input, outputs=x)
        return model

    @abstractmethod
    def _train(self):
        """Trains model"""
        raise NotImplementedError

    @abstractmethod
    def _build(self):
        """Builds the denoising model"""
        raise NotImplementedError

    @abstractmethod
    def scheduler(self):
        """Returns scheduler for model"""
        raise NotImplementedError

    @abstractmethod
    def optimizer(self):
        """Returns optimizer for model"""
        raise NotImplementedError

    def _training_keys(self) -> List:
        """Returns keys necessary for training configuration"""
        return ['epochs', 'loss_function', 'num_training_images', 'patch_size', 'batch_size',
                'grayscale', 'test', 'test_task', 'train', 'train_task', 'sigma', 'training_type']

    def _check_keys(self,
                    config: Dict,
                    keys: List,
                    exceptions: List = None):
        """Checks that model has appropriate configuration

        Parameters
        ----------
        config: (Dict) - Denoising Model configuration.
        keys: (List) - List of keys to check that config has.
        exceptions: (exceptions) - List of keys. If model does not have a key in keys, but key is in exceptions,
            then an exception will not be raised.
        """
        for key in keys:
            if key not in config:
                if not key in exceptions:
                    raise Exception(f'Model configuration missing key {key}')

        for key in config:
            if key not in keys:
                if not key in exceptions:
                    raise Exception(f'Model configuration should not contain key {key}')

    def _load_model(self, epochs: int) -> training.Model:
        """Loads or initializes model based on last trained epoch

        Parameters
        ----------
        epochs: (int) - Loads model based on number of epochs it was trained at

        Returns
        -------
        training.Model: Keras model

        """
        model = self._build()
        model = self._model(model)
        model.compile(optimizer=self.optimizer(), loss=retrieve_loss_function(self.config))
        if epochs == 0:
            return model
        else:
            file = os.path.join(self.data_dir, f'model_{epochs:04d}')
            loaded_model = load_model(file, custom_objects={self.config['loss_function']: retrieve_loss_function(self.config)})
            model.set_weights(loaded_model.get_weights())
            model.optimizer.set_weights(loaded_model.optimizer.get_weights())
            return model

    def _standard_training(self, save_every: int, test_every: int):
        """Standard training for denoising model.
        Simply trains using scheduler/optimizer for set number of epochs."""
        print(self.model.summary())
        print('Starting Standard Training')
        checkpointer = ModelCheckpoint(os.path.join(self.data_dir, 'model_{epoch:04d}'),
                                       verbose=1, save_weights_only=False, period=save_every)
        loss_file = os.path.join(self.model_dir, 'loss_log.csv')
        csv_logger = CSVLogger(loss_file, append=True, separator=',')
        lr_scheduler = LearningRateScheduler(self.scheduler())
        tester = PSNRTest(self, test_every=test_every)
        data_generator = DataGenerator(self.config, load_training_data=True, model_type=type(self))
        self.model.fit(data_generator,
                       epochs=self.config['epochs'], verbose=1,
                       callbacks=[lr_scheduler, checkpointer, csv_logger, tester],
                       initial_epoch=self.epochs)

    def _find_last_checkpoint(self) -> int:
        """Finds last checkpoint that model was saved at.

        Returns
        -------
        int: Last epoch model was saved at
        """
        file_list = glob.glob(os.path.join(self.data_dir, 'model_*'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*model_(.*)", file_)
                epochs_exist.append(int(result[0]))
            epoch = max(epochs_exist)
        else:
            epoch = 0
        return epoch

    def _mutable_config_keys(self):
        """Returns list of config model keys that are mutable and should not be saved in base configuration file.

        Returns
        -------
        (List) - List of mutable config keys.

        """
        return ['epochs', 'test', 'test_task']

    def _config_is_equal(self, config):
        """Checks if config is equivalent to this model config.

        Parameters
        ----------
        config: (Dict) - Configuration to be checked against.

        Returns
        -------
        (bool) - Returns True if config is equivalent to this model config.
        """
        for key in self.config:
            if key not in self._mutable_config_keys():
                if key not in config:
                    return False
                elif config[key] != self.config[key]:
                    return False
        for key in config:
            if key not in self._mutable_config_keys():
                if key not in self.config:
                    return False
                elif config[key] != self.config[key]:
                    return False
        return True

    def _save_config(self,
                     directory: str):
        """Saves config to model directory

        Parameters
        ----------
        directory: (str) - Directory that model will be saved to.
        """
        with open(os.path.join(directory, 'config.yaml'), 'w') as config_file:
            # Remove configurations we don't want saved
            values = [self.config.pop(key) for key in self._mutable_config_keys()]
            yaml.dump(self.config, config_file)
            for i, key in enumerate(self._mutable_config_keys()):
                self.config[key] = values[i]

    def _read_config(self,
                     directory: str) -> Dict:
        """Reads config from directory and returns it

        Parameters
        ----------
        directory: (str) - Directory to read config from

        Returns
        -------
        Dict: Config read from directory

        """
        with open(os.path.join(directory, 'config.yaml'), 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config

    def _find_model_dir(self) -> str:
        """Finds model directory. If not found, creates a new one.

        Returns
        -------
        str: Directory where model will be stored

        """
        for model_dir in os.listdir(self.models_dir):
            config = self._read_config(os.path.join(self.models_dir, model_dir))
            if self._config_is_equal(config):
                return os.path.join(self.models_dir, model_dir)
        new_dir = None
        for i in range(len(os.listdir(self.models_dir))):
            if not os.path.isdir(f'{self.models_dir}/model_{i:04d}'):
                new_dir = f'{self.models_dir}/model_{i:04d}'
                break
        if new_dir is None:
            new_dir = os.path.join(self.models_dir, f'model_{len(os.listdir(self.models_dir)):04d}')
        os.mkdir(new_dir)
        self._save_config(new_dir)
        return new_dir

    def test(self):
        """Tests the model on the specified test dataset and return dictionary of PSNR and RMSE results."""

        # Initialize test data
        data_generator = DataGenerator(self.config, load_training_data=False)

        # Get data range
        data_range = 1.0
        if self.config['train_task'] == 'denoise_curvature':
            data_range = 4+2*np.sqrt(2)

        # test the model
        rmses = []
        psnrs = []
        for noisy, clean in data_generator.generate_test_set():
            denoised = self.model.predict(noisy)
            rmses.append(rmse(denoised, clean))
            psnrs.append(psnr(denoised, clean, data_range=data_range))

        # Return metric values
        values = {'PSNR': np.mean(psnrs), 'RMSE': np.mean(rmses)}
        return values
