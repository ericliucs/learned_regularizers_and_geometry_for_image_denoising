from util import getGPU
from reproduce.thesis.train_models.curvature_approx_recon_models import curvature_approx_recon_models
import tensorflow as tf
import os
import csv


if __name__ == '__main__':
    getGPU()

    output_file = os.path.join(os.getcwd(), 'reproduce/thesis/learning_geometry/results/tnrd_approx_curvature_recon_results.csv')
    list_of_configs, model = curvature_approx_recon_models()
    psnr_vals = []
    for config in list_of_configs:
        tf.keras.backend.clear_session()
        name = config['R']['R']['name']
        if 'type' in config['R']['R']:
            name += config['R']['R']['type']
        psnr = model(config, add_keys=False, multitraining=False).test()['PSNR']
        psnr_vals.append([name, config['R']['R']['kernel_size'], config['R']['R']['filters'], psnr])
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Im68 Dataset'])
        writer.writerow(['Model', 'Filters', 'Avg PSNR'])
        for row in psnr_vals:
            writer.writerow(row)