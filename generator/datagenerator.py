import numpy as np
from tensorflow import keras
import os
import glob
from PIL import Image
from typing import Dict, List
import random


class DataGenerator(keras.utils.Sequence):
    '''Generates training and test data for Keras'''

    def __init__(self, config: Dict,
                 load_training_data: bool = True):
        """Initializes the data generator with the configuration

        Parameters
        ----------
        config: (Dict) - Dictionary of training and testing configuration
        load_training_data: (bool) - If True, training data will be loaded in preparation for Training.
        """
        self.config = config
        self.training_image_data = []
        self.testing_image_data = []
        # Get training data if specified
        if load_training_data:
            self.training_image_data = self._load_data(os.path.join('data', 'train', config['train']),
                                                       num=self.config['R'],
                                                       grayscale=self.config['grayscale'])
            self.training_image_data = self._convert_data_based_on_task(self.training_image_data,
                                                                        task=self.config['train_task'])
        # Get test data
        self.testing_image_data = self._load_data(os.path.join('data', 'test', config['test']),
                                                  grayscale=self.config['grayscale'])
        self.testing_image_data = self._convert_data_based_on_task(self.testing_image_data,
                                                                   task=self.config['test_task'])

    @staticmethod
    def _load_data(files_location: str,
                    num: int = None,
                    grayscale: bool = True) -> List[np.ndarray]:
        """Loads at most num data from files_location, converts to grayscale if specified, standardizes to [0,1]

        Parameters
        ----------
        files_location: (str) - Directory where data is stored
        num: (int) - Max number of images to load in
        grayscale: (bool) - If True, converts images to grayscale.

        Returns
        -------
        List[np.ndarray]: List of numpy images
        """
        file_list = glob.glob(files_location + '/*.png')
        file_list += glob.glob(files_location + '/*.jpg')
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
    def _convert_data_based_on_task(data: List[np.ndarray], task: str) -> List[np.ndarray]:
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
        else:
            raise NotImplementedError

    def _data_aug(self, img: np.ndarray, mode: int) -> np.ndarray:
        """Augments image by flipping or rotating. Code from
        https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/data_generator.py

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
        '''Returns number of batches per epoch'''
        # Will do number of distinct patches across all images
        num_pixels = 0
        for img in self.training_image_data:
            num_pixels += img.shape[0]*img.shape[1]
        batches_per_epcoh = num_pixels // (self.config['patch_size']**2)
        return 1000

    def __getitem__(self, index):
        '''Generate one batch of data'''
        y = np.asarray([self._extract_patch() for i in range(self.config['batch_size'])])
        X = y + np.random.normal(0, self.config['sigma']/255.0, y.shape)
        return X, y

    def generate_test_set(self):
        """Generator for test set
        """
        for y in self.testing_image_data:
            X = y + np.random.normal(0, self.config['sigma'] / 255.0, y.shape)
            yield X[np.newaxis, :, :, :], y[np.newaxis, :, :, :]
