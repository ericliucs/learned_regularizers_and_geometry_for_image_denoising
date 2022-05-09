from util import getGPU
from reproduce.thesis.train_models.denoised_curvature_recon_models import denoised_curvature_recon_models
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
    output_file = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/psnr_recon_results.csv')
    list_of_configs, model = denoised_curvature_recon_models()
    psnr_vals = []
    for config in list_of_configs:
        print(config)
        tf.keras.backend.clear_session()
        psnr = model(config, add_keys=False, multitraining=False).test()['PSNR']
        direct = True
        if config['loss_function'] == 'mean_sum_squared_error_curvature_loss':
            direct = False
        row = [get_full_model(config['F']), config['F']['R']['kernel_size'], get_direct(config['F'])]
        if 'D' in config:
            row += [get_full_model(config['D']), config['D']['R']['kernel_size'], get_direct(config['D']), psnr]
        else:
            row += [None, None, None, psnr]
        psnr_vals.append(row)
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Im68 Dataset'])
        writer.writerow(['F', 'Kernel Size', 'Direct', 'D', 'Kernel Size', 'Direct', 'Avg PSNR'])
        for row in psnr_vals:
                writer.writerow(row)