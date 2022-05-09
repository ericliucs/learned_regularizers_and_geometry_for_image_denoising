from matplotlib import pyplot as plt
import os
import csv


def read_csv_file(file):
    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        return list(reader)


if __name__ == '__main__':

    # Load rmse results
    rmse_results = os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results/denoising_curvature_results.csv')
    rmse_vals = read_csv_file(rmse_results)

    # Construct plot
    plt.clf()
    models = [(('False', 'True'), r'$\mathcal{F}^t_L(\kappa(f))$'),
          (('False', 'False'), r'$\kappa(\mathcal{F}^t_L(f))$'),
          (('True', 'True'), r'$\mathrm{TNRD}(\kappa(f))$'),
          (('True', 'False'), r'$(\kappa(\mathrm{TNRD}(f))$'),]
    # Full Model,Kernel Size,Direct,Avg RMSE
    for model, label in models:
        kernel_sizes = []
        rmses = []
        for row in rmse_vals:
            if row[0] == model[0] and row[2] == model[1]:
                kernel_sizes.append(int(row[1]))
                rmses.append(float(row[3]))
        plt.plot(kernel_sizes, rmses, label=label, marker='o', mfc='none')
    leg = plt.legend(loc = 'lower left', bbox_to_anchor=(0, .15, 0, 0))
    plt.grid()
    plt.xlabel('Kernel Size')
    plt.ylabel(r'RMSE')
    plt.savefig(os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results',
                             'rmse_denoising_curvature.png'))


    # Load psnr results
    plt.clf()
    psnr_results = os.path.join(os.getcwd(),
                                'reproduce/thesis/denoising_geometry/results/psnr_recon_results.csv')
    psnr_vals = read_csv_file(psnr_results)
    del psnr_vals[0]
    del psnr_vals[0]
    for model, label in models:
        kernel_sizes = []
        psnrs = []
        for row in psnr_vals:
            if row[0] == model[0] and row[2] == model[1] and row[4] == '7':
                kernel_sizes.append(int(row[1]))
                psnrs.append(float(row[6]))
        plt.plot(kernel_sizes, psnrs, label=label, marker='o', mfc='none')
    leg = plt.legend(loc = 'upper left')
    plt.grid()
    plt.xlabel('Kernel Size')
    plt.ylabel(r'PSNR')
    plt.savefig(os.path.join(os.getcwd(), 'reproduce/thesis/denoising_geometry/results',
                             'psnr_denoising_curvature_recon.png'))

    # Make table for G terms
    out_table = os.path.join(os.getcwd(),
                                'reproduce/thesis/denoising_geometry/results/denoisng_curvature_recon_G_table.csv')
    denoiser_settings = [('', 'f'), ('3', 'TNRD_3x3'), ('5', 'TNRD_5x5'), ('7', 'TNRD_7x7')]
    models = [(('True', '3'), 'Direct G_3x3'),
              (('False', '3'), 'Indirect G_3x3'),
              (('True', '5'), 'Direct G_5x5'),
              (('False', '5'), 'Indirect G_5x5'),
              (('True', '7'), 'Direct G_7x7'),
              (('False', '7'), 'Indirect G_7x7'),
              ]
    with open(out_table, 'w') as csv_file:
        writer = csv.writer(csv_file)
        model_names = [model[1] for model in models]
        writer.writerow(['D|F'] + model_names)
        for d_kernel_size, d_name in denoiser_settings:
            row = []
            row.append(d_name)
            for model in models:
                for temp_row in psnr_vals:
                    if temp_row[2] == model[0][0] and temp_row[1] == model[0][1] and temp_row[4] == d_kernel_size and temp_row[0] == 'False':
                        row.append(f'{float(temp_row[6]):.2f}')
            writer.writerow(row)

    # Make table for Full model
    out_table = os.path.join(os.getcwd(),
                             'reproduce/thesis/denoising_geometry/results/denoisng_curvature_recon_TNRD_table.csv')
    denoiser_settings = [('', 'f'), ('3', 'TNRD_3x3'), ('5', 'TNRD_5x5'), ('7', 'TNRD_7x7')]
    models = [(('True', '3'), 'Direct TNRD_3x3'),
              (('False', '3'), 'Indirect TNRD_3x3'),
              (('True', '5'), 'Direct TNRD_5x5'),
              (('False', '5'), 'Indirect TNRD_5x5'),
              (('True', '7'), 'Direct TNRD_7x7'),
              (('False', '7'), 'Indirect TNRD_7x7'),
              ]
    with open(out_table, 'w') as csv_file:
        writer = csv.writer(csv_file)
        model_names = [model[1] for model in models]
        writer.writerow(['D|F'] + model_names)
        for d_kernel_size, d_name in denoiser_settings:
            row = []
            row.append(d_name)
            for model in models:
                for temp_row in psnr_vals:
                    if temp_row[2] == model[0][0] and temp_row[1] == model[0][1] and temp_row[4] == d_kernel_size and \
                            temp_row[0] == 'True':
                        row.append(f'{float(temp_row[6]):.2f}')
            writer.writerow(row)
