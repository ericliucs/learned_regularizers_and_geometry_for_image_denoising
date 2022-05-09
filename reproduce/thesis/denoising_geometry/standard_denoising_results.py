from util import getGPU
from reproduce.thesis.train_models.standard_denoisers import standard_denoiser_configs
import tensorflow as tf
import os
import csv


if __name__ == '__main__':
    getGPU()
    output_file = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/denoising_results.csv')
    psnr_vals = []
    list_of_configs, model = standard_denoiser_configs()
    for config in list_of_configs:
        print(config)
        tf.keras.backend.clear_session()
        psnr = model(config, add_keys=False, multitraining=False).test()['PSNR']
        psnr_vals.append([config['R']['kernel_size'], psnr])
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Im68 Dataset'])
        writer.writerow(['Kernel Size','Avg psnr'])
        for row in psnr_vals:
                writer.writerow(row)