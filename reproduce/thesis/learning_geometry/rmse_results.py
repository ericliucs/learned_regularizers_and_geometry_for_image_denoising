from util import getGPU
from reproduce.thesis.train_models.curvature_approx_models import curvature_approx_models
import tensorflow as tf
import os
import csv


if __name__ == '__main__':
    getGPU()
    output_file = os.path.join(os.getcwd(), 'reproduce/thesis/learning_geometry/approx_curvature_results.csv')
    list_of_configs, model = curvature_approx_models()
    rmse_vals = []
    for config in list_of_configs:
        print(config)
        tf.keras.backend.clear_session()
        name = config['R']['name']
        if 'type' in config['R']:
            name += config['R']['type']
        rmse = model(config, add_keys=False, multitraining=False).test()['RMSE']
        rmse_vals.append([name, config['R']['kernel_size'], config['R']['filters'], rmse])
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Im68 Dataset'])
        writer.writerow(['Model', 'Filters', 'Avg RMSE'])
        for row in rmse_vals:
                writer.writerow(row)

