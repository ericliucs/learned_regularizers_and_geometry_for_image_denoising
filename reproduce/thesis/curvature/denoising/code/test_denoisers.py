from typing import List
from reproduce.thesis.curvature.denoising.code.curvature_denoisers import curvature_denoisers
from denoising.models.cnns.regularizers.regularizer_model import RegularizerModel
import csv
from denoising.util import clear_all_model_data
import os
from denoising.util import getGPU


def test_and_save_results(list_model_types_and_configs: List,
                          save_file: str):
    """Tests and saves results for the curvature denoisers"""
    getGPU()
    rmse_vals = {'TNRD_Direct': [],
                 'TNRD_Indirect': [],
                 'F_Direct': [],
                 'F_Indirect': [],
    }

    kernel_sizes = []
    for model_type, config in list_model_types_and_configs:
        model = model_type(config.copy(), add_keys=False)
        metrics = model.test()
        print(config)
        if model_type == RegularizerModel:
            if 'curvature' in config['loss_function']:
                rmse_vals['F_Indirect'].append((config['R']['kernel_size'], metrics['RMSE']))
            else:
                rmse_vals['F_Direct'].append((config['R']['kernel_size'], metrics['RMSE']))
        else:
            if 'curvature' in config['loss_function']:
                rmse_vals['TNRD_Indirect'].append((config['R']['kernel_size'], metrics['RMSE']))
            else:
                rmse_vals['TNRD_Direct'].append((config['R']['kernel_size'], metrics['RMSE']))
        if config['R']['kernel_size'] not in kernel_sizes:
            kernel_sizes.append(config['R']['kernel_size'])
        clear_all_model_data(model)
    kernel_sizes.sort()

    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Model'] + kernel_sizes)
        for key,item in rmse_vals.items():
            writer.writerow([key] + [val[1] for val in sorted(item, key=lambda x: x[0])])


if __name__ == '__main__':
    test_and_save_results(curvature_denoisers(),
                      os.path.join(os.getcwd(),
                                   'reproduce/thesis/curvature/denoising/results/spreadsheets/test_vals.csv'))
