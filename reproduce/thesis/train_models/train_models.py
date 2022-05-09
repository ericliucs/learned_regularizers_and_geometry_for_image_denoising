from reproduce.thesis.train_models.standard_denoisers import standard_denoiser_configs
from reproduce.thesis.train_models.curvature_approx_models import curvature_approx_models
from reproduce.thesis.train_models.curvature_approx_recon_models import curvature_approx_recon_models
from reproduce.thesis.train_models.curvature_denoisers import all_curvature_denoisers
from reproduce.thesis.train_models.denoised_curvature_recon_models import denoised_curvature_recon_models
from reproduce.thesis.train_models.denoised_curvature_recon_stages import denoised_curvature_recon_stages_models
from util import getGPU
import tensorflow as tf
import os


# Train standard TNRD, DnCNN, TDV

# Train approximate regularizers
# - TNRD, MTNRD, ML2TNRD for small weights

# - Full TNRD, Full MTNRD, Full ML2TNRD, DnCNNDirect, TDV

# Train oracle reconstruction; will simply set up a reconstruction process to do this
# - Train for all regularizers above


#########################################################
# Train curvature denoisers

# Train reconstruction with curvature denoisers

# Train boosting reconstructions with curvature denoisers

#######################################################################


# Train GF models

## Train GF-TNRD Models

## Train GF-TDV models

def is_tdv(config):
    for keys, value in config.items():
        if isinstance(value, dict):
            if is_tdv(value):
                return True
        else:
            if value == 'TDV':
                return True
    return False

def is_dncnn(config):
    for keys, value in config.items():
        if isinstance(value, dict):
            if is_dncnn(value):
                return True
        else:
            if value == 'DnCNN':
                return True
    return False


def add_len(config):
    config['len_test_1'] = True
    for keys, value in config.items():
        if isinstance(value, dict):
            if 'R' in value:
                add_len(value)

def replace_all_epochs_with_zero(config):
    for key in config:
        if key == 'epochs':
            config[key] = 0
        if isinstance(config[key], dict):
            replace_all_epochs_with_zero(config[key])

            

if __name__ == '__main__':
    getGPU()
    test = False

    trainings = [curvature_approx_recon_models()]

    # trainings = [
    #    standard_denoiser_configs(), curvature_approx_models(), curvature_approx_recon_models(), denoised_curvature_recon_stages_models()
    # ]
    # trainings += all_curvature_denoisers() + [denoised_curvature_recon_models()]

    fail_file = os.path.join(os.getcwd(), 'reproduce/thesis/train_models/failures.txt')
    open(fail_file, 'w').close()

    for list_of_configs, model in trainings:
        for config in list_of_configs:
            tf.keras.backend.clear_session()
            print(config)
            if test:
                if not is_tdv(config):
                    if not is_dncnn(config):
                        print(config)
                        add_len(config)
                        model(config, add_keys=False, multitraining=False)
            else:
                #model(config, add_keys=False, multitraining=False)
                psnr_val = model(config, add_keys=False, multitraining=False).test()['PSNR']
                print(psnr_val)
                # try:
                #     #replace_all_epochs_with_zero(config)
                #     if config['S'] < 8:
                #         this_model = model(config, add_keys=False, multitraining=False)
                #         psnr_val = this_model.test()['PSNR']
                #         print(psnr_val)
                #     # weights = this_model.get_weights()
                #     # print(weights[0][:,:,0,0])
                #     # print(weights[1][:,:,0,0])
                # except Exception:
                #     print('Training Failed')
                #     with open(fail_file, 'a') as txt_file:
                #         txt_file.write('Failed training a model\n')
