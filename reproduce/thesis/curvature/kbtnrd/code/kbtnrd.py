
from denoising.util import getGPU
from reproduce.thesis.curvature.kbtnrd.code.configs import model_types_and_configs
from typing import List
import csv, os


def test_and_save_results(list_model_types_and_configs: List,
                          save_file: str):
    """Tests and save results for KB-TNRD models"""
    getGPU()
    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Filter Size', 'KB-TNRD'])
        for model_type, config in list_model_types_and_configs:
            model = model_type(config.copy(), add_keys=False)
            metrics = model.test()
            writer.writerow([config['F']['R']['kernel_size'],metrics['PSNR']])


if __name__ == '__main__':
    getGPU()
    list_of_model_types_and_configs = model_types_and_configs()
    test_and_save_results(list_of_model_types_and_configs,
                          os.path.join(os.getcwd(),
                                       'reproduce/thesis/curvature/kbtnrd/results/spreadsheets/test_psnr_vals.csv'))
