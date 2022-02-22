from abc import ABC, abstractmethod
from typing import Dict
import os
import yaml
import glob
import re
from tensorflow.keras.models import load_model
from tensorflow.python.keras.engine import training
from training.training import retrieve_training_function
from training.losses import retrieve_loss_function


class CNNModel(ABC):
    """Defines abstract class for saved_models"""

    def __init__(self,
                 config: Dict,
                 verbose: bool = False,
                 train: bool=True):

        if verbose:
            def log(x): print(x)
        else:
            def log(x): pass
        self.log = log

        self.config = config
        self._add_keys_to_config()
        log(f'Config set: {self.config}')
        self._check_config_for_model_keys()
        self._check_config_for_training_keys()
        self.models_dir = os.path.join(os.getcwd(), 'models', 'saved_models')
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)
        self.model_dir = self._find_model_dir()

        self.epochs = self._find_last_checkpoint()
        self.log(f'Last checkpoint found: {self.epochs}')
        # Load model and train if not trained
        if self.epochs >= self.config['epochs']:
            self.model = self._load_model(self.config['epochs'])
        else:
            self.model = self._load_model(self.epochs)
            self.log('Training Model')
            if train:
                self._train()
                self.epochs = self.config['epochs']

    def summary(self):
        print(self.model.summary())

    @abstractmethod
    def _check_config_for_model_keys(self):
        """Checks that config has all the necessary keys for model"""

    @abstractmethod
    def _add_keys_to_config(self):
        """Adds keys to config that are always the same"""

    def _check_config_for_training_keys(self):
        """Checks that config has all other necessary keys for training"""
        pass

    @abstractmethod
    def scheduler(self):
        """Returns scheduler for model"""

    @abstractmethod
    def optimizer(self):
        """Returns optimizer for model"""

    @abstractmethod
    def _model(self):
        """Defines Keras model"""

    @abstractmethod
    def visualize_parameters(self):
        """"Visualizes model parameters"""

    def _train(self):
        """Trains model"""
        retrieve_training_function(self.config)(self)

    def test(self):
        """Tests model on test set from generator"""
        raise NotImplementedError

    def _load_model(self, epochs: int) -> training.Model:
        """Loads or initializes model based on last trained epoch

        Parameters
        ----------
        epochs: (int) - Loads model based on number of epochs it was trained at

        Returns
        -------
        training.Model: Keras model

        """
        if epochs == 0:
            return self._model()
        else:
            file = os.path.join(self.model_dir, f'model_{epochs:04d}')
            return load_model(file, custom_objects={self.config['loss_function']: retrieve_loss_function(self.config)})

    def _find_last_checkpoint(self) -> int:
        """Finds last checkpoint that model was saved at.

        Returns
        -------
        int: Last epoch model was saved at
        """
        file_list = glob.glob(os.path.join(self.model_dir, 'model_*'))
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
        return ['epochs', 'test', 'test_task', 'checkpoint']

    def _config_is_equal(self, config):
        """Checks if two configs are equal"""
        for key in config:
            if key not in self._mutable_config_keys():
                if key not in self.config:
                    return False
                elif config[key] != self.config[key]:
                    return False
        return True

    def _save_config(self, directory: str):
        """Saves config to model directory"""
        with open(os.path.join(directory, 'config.yaml'), 'w') as config_file:
            # Remove configurations we don't want saved
            values = [self.config.pop(key) for key in self._mutable_config_keys()]
            yaml.dump(self.config, config_file)
            self.log(f'Writing config to yaml file in: {directory}')
            for i, key in enumerate(self._mutable_config_keys()):
                self.config[key] = values[i]

    def _read_config(self, directory: str) -> Dict:
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
        self.log(f'Reading config from yaml file: {config}')
        return config

    def _find_model_dir(self) -> str:
        """Finds model directory. If not found, creates a new one.

        Returns
        -------
        str: Directory where model will be stored

        """
        self.log("Searching for model directory")
        for model_dir in os.listdir(self.models_dir):
            config = self._read_config(os.path.join(self.models_dir, model_dir))
            if self._config_is_equal(config):
                self.log(f"Found model directory: {model_dir}")
                return os.path.join(self.models_dir, model_dir)
        new_dir = os.path.join(self.models_dir, f'model_{len(os.listdir(self.models_dir)):04d}')
        os.mkdir(new_dir)
        self._save_config(new_dir)
        self.log(f"Creating new model directory: {new_dir}")
        return new_dir
