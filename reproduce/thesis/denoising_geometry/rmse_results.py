from util import getGPU
from reproduce.thesis.train_models.curvature_denoisers import all_curvature_denoisers
import tensorflow as tf
import os
import csv


def get_direct(config):
    if config['loss_function'] == 'mean_sum_squared_error_curvature_loss':
        return False
    else:
        return True


def get_full_model(config):
    if 'S' in config:
        return True
    else:
        return False

if __name__ == '__main__':
    getGPU()
    output_file = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/denoising_curvature_results.csv')
    rmse_vals = []
    for list_of_configs, model in all_curvature_denoisers():
        for config in list_of_configs:
            print(config)
            tf.keras.backend.clear_session()
            rmse = model(config, add_keys=False, multitraining=False).test()['RMSE']
            rmse_vals.append([get_full_model(config), config['R']['kernel_size'], get_direct(config), rmse])
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Im68 Dataset'])
        writer.writerow(['Full Model', 'Kernel Size', 'Direct', 'Avg RMSE'])
        for row in rmse_vals:
                writer.writerow(row)

