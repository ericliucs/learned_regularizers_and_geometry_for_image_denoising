from matplotlib import pyplot as plt
import os
import csv


def read_csv_file(file):
    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        return list(reader)


if __name__ == '__main__':

    # Load rmse results
    rmse_results = os.path.join(os.getcwd(), 'reproduce/thesis/learning_geometry/results/approx_curvature_results.csv')
    rmse_vals = read_csv_file(rmse_results)

    # Construct plot
    plt.clf()
    gs = [('TNRD', r'$\mathcal{G}_{L_1}$'), ('KTNRDstandard', r'$\mathcal{G}_{L_2}$'),
          ('KTNRDL1', r'$\mathcal{G}_{L_3}$'), ('KTNRDL2', r'$\mathcal{G}_{L_4}$')]
    for model, label in gs:
        filters = []
        rmses = []
        for row in rmse_vals:
            if row[0] == model and row[1] == '3' and row[2] != '8':
                filters.append(int(row[2]))
                rmses.append(float(row[3]))
        plt.plot(filters, rmses, label=label, marker='o', mfc='none')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of Filters')
    plt.ylabel('RMSE')
    plt.savefig(os.path.join(os.getcwd(), 'reproduce/thesis/learning_geometry/results', 'rmse_curvature_approx.png'))

    recon_results = os.path.join(os.getcwd(),
                               'reproduce/thesis/learning_geometry/results/tnrd_approx_curvature_recon_results.csv')
    recon_vals = read_csv_file(recon_results)

    # Construct plot
    plt.clf()
    for model, label in gs:
        filters = []
        psnrs = []
        for row in recon_vals:
            if row[0] == model and row[1] == '3' and row[2] != '8':
                filters.append(int(row[2]))
                psnrs.append(float(row[3]))
        plt.plot(filters, psnrs, label=label, marker='o', mfc='none')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of Filters')
    plt.ylabel('PSNR')
    plt.savefig(os.path.join(os.getcwd(), 'reproduce/thesis/learning_geometry/results', 'psnr_curvature_approx_recon.png'))

    plt.clf()
    del recon_vals[0]
    del recon_vals[0]
    sizes = ['3', '5', '7']
    kernel_sizes = []
    psnrs = []
    for size in sizes:
        for row in recon_vals:
            print(row)
            if row[1] == size:
                if row[1] == '3':
                    if not (row[2] == '1' or row[2] == '2' or row[2] == '48'):
                        kernel_sizes.append(int(row[1]))
                        psnrs.append(float(row[3]))
                else:
                    kernel_sizes.append(int(row[1]))
                    psnrs.append(float(row[3]))
    plt.plot(kernel_sizes, psnrs, label=r'$\mathcal{G}_{L_1}}$', marker='o', mfc='none')
    plt.legend()
    plt.grid()
    plt.xlabel('Kernel Size')
    plt.ylabel('PSNR')
    plt.savefig(
        os.path.join(os.getcwd(), 'reproduce/thesis/learning_geometry/results', 'psnr_curvature_approx_recon_full.png'))