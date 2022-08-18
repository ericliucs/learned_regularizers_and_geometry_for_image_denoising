from denoising.models.cnns.gfvnet import GFVNetModel

def base_config():
    config = {}

    # Training settings
    config['training_type'] = 'standard'
    config['epochs'] = 10
    config['loss_function'] = 'mean_sum_squared_error_loss'
    config['num_training_images'] = 400
    config['patch_size'] = 50
    config['batch_size'] = 64
    config['grayscale'] = True
    config['test'] = 'BSDS68'
    config['test_task'] = 'denoising'
    config['train'] = 'BSDS400'
    config['train_task'] = 'denoising'
    config['sigma'] = 25

    # GFVNet Process
    config['constant_dataterm_weight'] = False
    config['scale_dataterm_weight'] = False
    config['regularizer_weight'] = False
    config['scale_regularizer_weight'] = False
    config['constant_regularizer_weight'] = False
    config['constant_geometry'] = False
    config['constant_denoiser'] = False
    config['use_prox'] = False
    config['S'] = 1
    config['direct_geometry_denoising'] = True
    config['steps_per_geometry'] = 1
    config['use_recon'] = False


    config['G'] = {'name': 'TNRD',
                       'filters': 8,
                       'kernel_size': 3,
                       'grayscale': True}

    config['G'] = {'name': 'TNRD',
                   'filters': 8,
                   'kernel_size': 3,
                   'grayscale': True}

    config['F'] = {'name': 'TNRD',
                   'filters': 8,
                   'kernel_size': 3,
                   'grayscale': True}
    return config


def list_of_model_types_and_configs():

    # Define model settings that will be tested
    kernel_size_and_filter_nums = [(3,8), (5, 24), (7, 48), (9, 80)]
    sigmas = [15,25,50]
    S = [3,5,7]

    # Generate all configs
    model_types_and_configs = []
    for sigma in sigmas:
        for s in S:
            for g_kernel_size, g_filter_num in kernel_size_and_filter_nums:
                for f_kernel_size, f_filter_num in kernel_size_and_filter_nums:
                    config = base_config()
                    config['sigma'] = sigma
                    config['G']['kernel_size'] = g_kernel_size
                    config['G']['filters'] = g_filter_num
                    config['F']['kernel_size'] = f_kernel_size
                    config['F']['filters'] = f_filter_num
                    config['S'] = s
                    model_types_and_configs.append((GFVNetModel, config))

    return model_types_and_configs
