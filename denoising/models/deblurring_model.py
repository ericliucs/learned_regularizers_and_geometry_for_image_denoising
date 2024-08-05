from abc import ABC, abstractmethod
import os
import yaml

class DeblurringModel(ABC):
    """Defines the interface for deblurring models."""

    def __init__(self, config: dict, should_train: bool = True, multitraining: bool = False):
        """Initializes the denoising model

                Parameters
                ----------
                config: (Dict) - Configuration settings for model
                should_train: (bool) - If True, model will be trained if it has not already been
                multitraining: (bool) - Specify as true if training the models across multiple computers which are
                    using the same memory space to store the results.
                """
        self.config = config

        # Find or create model directory
        self.models_dir = os.path.join(os.getcwd(), 'denoising', 'models', 'saved_models')
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)
        self.model_dir = self._find_model_dir()
        self.data_dir = os.path.join(self.model_dir, 'data')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        print(self.data_dir)

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

    def _read_config(self,
                     directory: str) -> dict:
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

    # TODO Is this method even necessary in the original code? If the configs are dicts, then Python can just compare...
    def _config_is_equal(self, config):
        """Checks if config is equivalent to this model config.

        Parameters
        ----------
        config: (Dict) - Configuration to be checked against.

        Returns
        -------
        (bool) - Returns True if config is equivalent to this model config.
        """
        return self.config == config

    def _mutable_config_keys(self):
        """Returns list of config model keys that are mutable and should not be saved in base configuration file.

        Returns
        -------
        (List) - List of mutable config keys.

        """
        return ['epochs', 'test', 'test_task']
