
from denoising.util import clear_all_model_data, getGPU
from reproduce.thesis.curvature.recon.code.configs import model_types_and_configs
from typing import List
import csv
import os


def test_and_save_results(list_model_types_and_configs: List,
                          save_file: str):
    """Tests and saves results of reconstructing from denoised curvature"""

    getGPU()
    psnr_vals = {'Recon-Trainable': [],
                 'Recon-Non-Trainable': [],
    }

    stages  = []
    for model_type, config in list_model_types_and_configs:
        model = model_type(config.copy(), add_keys=False)
        metrics = model.test()
        if config['F_train']:
            psnr_vals['Recon-Trainable'].append((config['S'], metrics['PSNR']))
        else:
            psnr_vals['Recon-Non-Trainable'].append((config['S'], metrics['PSNR']))
        if config['S'] not in stages:
            stages.append(config['S'])
        clear_all_model_data(model)
        clear_all_model_data(model)
    stages.sort()

    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['TNRD Recon Model'] + stages)
        for key,item in psnr_vals.items():
            writer.writerow([key] + [val[1] for val in sorted(item, key=lambda x: x[0])])


if __name__ == '__main__':
    getGPU()
    list_of_model_types_and_configs = model_types_and_configs()
    test_and_save_results(list_of_model_types_and_configs,
                          os.path.join(os.getcwd(),
                                       'reproduce/thesis/curvature/recon/results/spreadsheets/recon_denoise_curvature.csv'))
