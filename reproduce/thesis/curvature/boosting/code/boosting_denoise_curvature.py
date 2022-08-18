
from denoising.util import clear_all_model_data, getGPU
from reproduce.thesis.curvature.boosting.code.configs import model_types_and_configs
from typing import List
import csv, os


def test_and_save_results(list_model_types_and_configs: List,
                          save_file: str):
    """Tests and saves the results of the boosted models"""
    getGPU()
    psnr_vals = {'D_NTrain_F_NTrain': [],
                 'D_Train_F_NTrain': [],
                 'D_NTrain_F_Train': [],
                 'D_Train_F_Train': [],
    }

    stages  = []
    for model_type, config in list_model_types_and_configs:
        model = model_type(config.copy(), add_keys=False)
        metrics = model.test()
        if config['F_train']:
            if config['D_train']:
                psnr_vals['D_Train_F_Train'].append((config['S'], metrics['PSNR']))
            else:
                psnr_vals['D_NTrain_F_Train'].append((config['S'], metrics['PSNR']))
        else:
            if config['D_train']:
                psnr_vals['D_Train_F_NTrain'].append((config['S'], metrics['PSNR']))
            else:
                psnr_vals['D_NTrain_F_NTrain'].append((config['S'], metrics['PSNR']))
        if config['S'] not in stages:
            stages.append(config['S'])
        clear_all_model_data(model)
    stages.sort()

    with open(save_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Boosting Model'] + stages)
        for key,item in psnr_vals.items():
            writer.writerow([key] + [val[1] for val in sorted(item, key=lambda x: x[0])])


if __name__ == '__main__':
    getGPU()
    test_and_save_results(model_types_and_configs(),
                          os.path.join(os.getcwd(),
                                       'reproduce/thesis/curvature/boosting/results/spreadsheets/boosting_denoise_curvature.csv'))
