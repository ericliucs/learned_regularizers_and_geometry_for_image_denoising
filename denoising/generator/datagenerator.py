import numpy as np
from tensorflow import keras
import os
import glob
from PIL import Image
import random
from typing import Dict, Union, List
from denoising.models.cnns.regularizers.tv.tv import TVRegularizer


class DataGenerator(keras.utils.Sequence):
    """Generates training and test data for Keras"""

    def __init__(self, config: Dict,
                 load_training_data: bool = True):
        """Initializes the data generator with the configuration

        Parameters
        ----------
        config: (Dict) - Dictionary of training and testing configuration
        load_training_data: (bool) - If True, training data will be loaded in preparation for Training.
        """
        self.config = config

        # Get training data if specified
        if load_training_data and 'train' in config:
            self.training_image_data = self._load_data(self.config['train'],
                                                       num=self.config['num_training_images'],
                                                       grayscale=self.config['grayscale'])
            self.training_image_data = self._convert_data_based_on_task(self.training_image_data,
                                                                        task=self.config['train_task'])
        # Get test data
        if 'test' in config:
            self.testing_image_data = self._load_data(config['test'],
                                                      grayscale=self.config['grayscale'])
            self.testing_image_data = self._convert_data_based_on_task(self.testing_image_data,
                                                                       task=self.config['test_task'])

        if 'denoise_curvature' in self.config['train_task'] or 'approx_curvature' in self.config['train_task']:
            self.curvature = TVRegularizer({}).grad

    @staticmethod
    def _load_data(files_location: Union[str, List],
                   num: int = None,
                   grayscale: bool = True) -> List[np.ndarray]:
        """Loads at most num data from files_location, converts to grayscale if specified, standardizes to [0,1]

        Parameters
        ----------
        files_location: (Union[str, List]) - Directory  or list of directories where data is stored
        num: (int) - Max number of images to load in
        grayscale: (bool) - If True, converts images to grayscale.

        Returns
        -------
        List[np.ndarray]: List of numpy images
        """
        if isinstance(files_location, str):
            files_location = [files_location]
        file_list = []
        for loc in files_location:
            location = os.path.join('data', loc)
            file_list += glob.glob(location + '/*.png')
            file_list += glob.glob(location + '/*.jpg')
        file_list = sorted(file_list)
        file_list = file_list[:num]
        img_list = []
        for file in file_list:
            with Image.open(file) as im:
                img = np.asarray(im)
                if grayscale:
                    img = np.mean(img.astype('float32'), 2, keepdims=True) / 255.0
                else:
                    img = img.astype('float32') / 255.0
                img_list.append(img)
        return img_list

    @staticmethod
    def _data_aug(img: np.ndarray, mode: int) -> np.ndarray:
        """Augments image by flipping or rotating.

        Parameters
        ----------
        img: (np.ndarray) - Image to be augmented
        mode: (mode) - Mode of augmentation

        Returns
        -------
        np.ndarray: Augmented image

        """
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(img)
        elif mode == 2:
            return np.rot90(img)
        elif mode == 3:
            return np.flipud(np.rot90(img))
        elif mode == 4:
            return np.rot90(img, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 6:
            return np.rot90(img, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))

    def _extract_patch(self) -> np.ndarray:
        """Extracts random patch from data and augments it by flipping or rotating the patch

        Returns
        -------
        np.ndarray: Random augmented patch from training data

        """
        im = random.choice(self.training_image_data)
        offset_y = np.random.randint(0, im.shape[0] - self.config['patch_size'])
        offset_x = np.random.randint(0, im.shape[1] - self.config['patch_size'])
        patch = im[offset_y:offset_y + self.config['patch_size'], offset_x:offset_x + self.config['patch_size'], :]
        return self._data_aug(patch, mode=np.random.randint(0, 8))

    def __len__(self):
        """Returns number of batches per epoch"""
        if 'len_test_1' in self.config:
            return 1
        else:
            return 10000

    def _convert_data_based_on_task(self, data: List[np.ndarray], task: str) -> List[np.ndarray]:
        """Converts image data based on specified task

        Parameters
        ----------
        data: (List) - List of image data
        task: (task) - Model task. For example, denoising or super resolution.

        Returns
        -------
        List: Converted image data list

        """
        if task == 'denoising':
            return data
        elif 'denoise_curvature' in task:
            return data
        elif 'approx_curvature' in task:
            return data
        elif 'oracle_recon' in task:
            return data
        else:
            raise NotImplementedError

    def _generate_x_based_on_task(self, data, task):
        """Generates X data based on specified task
        """
        if task == 'denoising' or task == 'denoise_curvature' or task == 'oracle_recon':
            return data + np.random.normal(0, self.config['sigma']/255.0, data.shape).astype(np.float32)
        elif 'approx_curvature' in task:
            return data
        else:
            raise NotImplementedError

    def _convert_batch_based_on_task(self, x, y, task):
        """Converts batch x and y based on specified task"""
        if task == 'denoising':
            return x, y
        elif 'denoise_curvature' in task:
            return self.curvature(x), self.curvature(y)
        elif 'approx_curvature' in task:
            return x, self.curvature(y)
        elif 'oracle_recon' in task:
            return [x,y], y
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        """Generate one batch of data"""
        y = np.asarray([self._extract_patch() for _ in range(self.config['batch_size'])])
        X = self._generate_x_based_on_task(y, self.config['train_task'])
        return self._convert_batch_based_on_task(X, y, self.config['train_task'])

    def generate_test_set(self):
        """Generator for test set
        """
        for y in self.testing_image_data:
            X = self._generate_x_based_on_task(y, self.config['test_task'])
            yield self._convert_batch_based_on_task(X[np.newaxis, :, :, :],
                                                    y[np.newaxis, :, :, :], self.config['test_task'])
