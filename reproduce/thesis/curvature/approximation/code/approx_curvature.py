from typing import List
from denoising.models.cnns.regularizers.regularizer_model import RegularizerModel
import csv
import os
from denoising.util import clear_all_model_data


def approx_curvature_regularizer_models() -> List:
    """Returns list of model types and configurations that will be tested in terms of their ability to approximate
        curvature.

    Returns
    -------
    List of tuples of size 2. First entry of tuple is model type. Second entry of tuple
        is configuration of the model.

    """
    # Config settings that are common across all the models
    standard_approx_curvature_config = {
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
        'training_type': 'standard'
    }

    # Regularizer config settings to be tested
    reg_configs = [
        {'name': 'TNRD',
         'filters': 8,
         'kernel_size': 3,
         'grayscale': True},
        {'name': 'TNRD',
         'filters': 24,
         'kernel_size': 5,
         'grayscale': True},
        {'name': 'TNRD',
         'filters': 48,
         'kernel_size': 7,
         'grayscale': True},
    ]

    # Combine all settings
    configs = []
    for reg_config in reg_configs:
        config = standard_approx_curvature_config.copy()
        config['R'] = reg_config.copy()
        configs.append(config)
    # Return list
    return [(RegularizerModel, config) for config in configs]


def test_and_save_results(list_model_types_and_configs: List,
                          save_file: str,
                          col_headings: List = None,
                          stages: bool = False):
    """Tests and saves rmse values for curvature approximation models.

    Parameters
    ----------
    list_model_types_and_configs: (List) - List of tuples containing model type and config
    save_file: (str) - File to save results to
    col_headings: (List) - List of headings for three columns.
    """
    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        if col_headings is None:
            writer.writerow(['Model', 'Stages', 'RMSE', 'PSNR'])
        else:
            writer.writerow(col_headings)
        for model_type, config in list_model_types_and_configs:
            model = model_type(config.copy(), add_keys=False)
            metrics = model.test()
            print(config)
            if 'name' in config['R']:
                model_name = config['R']['name']
                if 'kernel_size' in config['R']:
                    kernel_size = config['R']['kernel_size']
            else:
                model_name = config['R']['R']['name']
                if 'kernel_size' in config['R']['R']:
                    kernel_size = config['R']['R']['kernel_size']
            if model_name == 'TNRD':
                model_name = f'{model_name}_{kernel_size}x{kernel_size}'
            if stages:
                writer.writerow([model_name, config['S'], metrics['RMSE'], metrics['PSNR']])
            else:
                writer.writerow([model_name, metrics['RMSE'], metrics['PSNR']])
            clear_all_model_data(model)


if __name__ == '__main__':
    test_and_save_results(approx_curvature_regularizer_models(),
                       os.path.join(os.getcwd(),
                                    'reproduce/thesis/curvature/approximation/results/spreadsheets/approx_curvature.csv'),
                          col_headings=['Model', 'RMSE(k(a), model(a))', 'PSNR(k(a), model(a))'])
