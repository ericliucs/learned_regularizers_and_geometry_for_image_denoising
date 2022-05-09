from util import getGPU
from reproduce.thesis.train_models.denoised_curvature_recon_stages import denoised_curvature_recon_stages_models
import tensorflow as tf
import os
import csv

if __name__ == '__main__':
    getGPU()
    output_file = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/psnr_recon_stages_results.csv')
    list_of_configs, model = denoised_curvature_recon_stages_models()
    psnr_vals = []
    for config in list_of_configs:
        print(config)
        tf.keras.backend.clear_session()
        psnr = model(config, add_keys=False, multitraining=False).test()['PSNR']
        if 'D' in config:
            row = ['D']
        else:
            row = ['f']
        row += [config['S'], psnr]
        psnr_vals.append(row)
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Im68 Dataset'])
        writer.writerow(['Input', 'Stages','Avg PSNR'])
        for row in psnr_vals:
                writer.writerow(row)