import os
import platform
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import pickle
from itertools import product
import numpy as np
import tabulate


if __name__ == '__main__':
    # General parameters
    n_of_workers = [4, 16]  # Number of workers
    modes = ['balanced', 'unbalanced']
    n_threads = 2
    epochs = 300

    # Train autoencoder
    order = 'autoencoders.py --latentdim=' + str(50) + ' --w_sent=' + str(50000) \
            + ' --intdim1=' + str(128) + ' --intdim2=' + str(64) + ' --task=getae'

    if platform.system() == 'Windows':
        print("Running on Windows")
        _ = os.system('python ' + order)  # Windows order
    else:
        print("Running on Linux")
        _ = os.system('python3 ' + order)  # Linux order

    # Baseline centralized
    order = 'baseline_executer.py --now=' + str(1) + ' --mode=' + str('balanced') + ' --epochs=' + str(epochs) \
            + ' --w_sent=' + str(50000)  # Baseline case
    if platform.system() == 'Windows':
        print("Running on Windows")
        _ = os.system('python ' + order)  # Windows order
    else:
        print("Running on Linux")
        _ = os.system('python3 ' + order)  # Linux order

    # Train all autoencoder cases and baseline cases
    def process_tr(now, mode):
        order1 = 'autoencoders.py --latentdim=' + str(50) + ' --w_sent=' + str(50000) + ' --intdim1=' + str(128) \
                 + ' --intdim2=' + str(64) + ' --task=train' + ' --now=' + str(now) + ' --mode=' + str(mode) \
                 + ' --epochs=' + str(epochs)

        order2 = 'baseline_executer.py --now=' + str(now) + ' --mode=' + str(mode) + ' --epochs=' + str(epochs) \
                 + ' --w_sent=' + str(50000)

        if platform.system() == 'Windows':
            print("Running on Windows")
            _ = os.system('python ' + order1)  # Windows order
            _ = os.system('python ' + order2)  # Windows order
        else:
            print("Running on Linux")
            _ = os.system('python3 ' + order1)  # Linux order
            _ = os.system('python3 ' + order2)  # Linux order

    _ = Parallel(n_jobs=n_threads, verbose=10) \
        (delayed(process_tr)(now=now, mode=mode) for now in n_of_workers for mode in modes)


    # Show results and save figure
    color_dict = {'1': 'r', '4': 'b', '16': 'k', '2': 'g', '8':'y'}  # Color is for the number of workers
    mark_dict = {'baseline': 'o', 'ae': 's'}  # Mark is for the type of result
    line_dict = {'balanced': '-', 'unbalanced': '--'}  # Line is for the mode
    x_ep = np.arange(epochs)
    n_of_workers.append(1)
    # Plot the mean loss values for all obtained data
    tab = np.zeros((2, len(n_of_workers), len(modes)))  # Type of result x now x mode
    for now, mode in product(n_of_workers, modes):
        # Obtain the AE results
        save_file = 'results_auto_' + str(now) + '_mode_' + str(mode) + '.pickle'
        try:
            with open(save_file, 'rb') as handle:
                data = pickle.load(handle)
        except:
            data['loss_global'] = np.zeros((epochs, now))
        tab[0, n_of_workers.index(now), modes.index(mode)] = np.mean(data['loss_global'][-1, :])
        plt.semilogy(x_ep, np.mean(data['loss_global'], axis=1), color=color_dict[str(now)],
                     marker=mark_dict['ae'], linestyle=line_dict[str(mode)],
                     label='NoW = ' + str(now) + ', mode = ' + str(mode) + ', AE')
        print('Losses: NoW = ', now, ', mode = ', mode, 'AE --> Final value = ',
              np.mean(data['loss_global'][-1, :]))
        # Obtain the baseline results
        save_file = 'results_baseline_' + str(now) + '_' + str(mode) + '.pickle'
        try:
            with open(save_file, 'rb') as handle:
                data = pickle.load(handle)
        except:
            data['loss_global'] = np.zeros((epochs, now))
        tab[1, n_of_workers.index(now), modes.index(mode)] = np.mean(data['loss_global'][-1, :])
        plt.semilogy(x_ep, np.mean(data['loss_global'], axis=1), color=color_dict[str(now)],
                     marker=mark_dict['baseline'], linestyle=line_dict[str(mode)],
                     label='NoW = ' + str(now) + ', mode = ' + str(mode) + ', baseline')
        print('Losses: NoW = ', now, ', mode = ', mode, 'baseline --> Final value = ',
              np.mean(data['loss_global'][-1, :]))
    plt.title('Global losses')
    plt.xlabel('Epoch')
    plt.ylabel('Mean global losses')
    # plt.legend(loc='best')
    tikz_save('loss_global.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
    plt.savefig('loss_global.png', bbox_inches='tight')

    for mode in modes:
        print('For ', mode)
        print(tabulate.tabulate(tab[:, :, modes.index(mode)], headers=n_of_workers, showindex=['ae', 'bas'], tablefmt='latex'))
    #plt.show()
    plt.close()

    # Plot the mean accuracy values for all obtained data
    tab = np.zeros((2, len(n_of_workers), len(modes)))  # Type of result x now x mode
    for now, mode in product(n_of_workers, modes):
        # Obtain the AE results
        save_file = 'results_auto_' + str(now) + '_mode_' + str(mode) + '.pickle'
        try:
            with open(save_file, 'rb') as handle:
                data = pickle.load(handle)
        except:
            data['accuracy_global'] = np.zeros((epochs, now))
        tab[0, n_of_workers.index(now), modes.index(mode)] = np.mean(data['accuracy_global'][-1, :])
        plt.semilogy(x_ep, 1 - np.mean(data['accuracy_global'], axis=1), color=color_dict[str(now)],
                     marker=mark_dict['ae'], linestyle=line_dict[str(mode)],
                     label='NoW = ' + str(now) + ', mode = ' + str(mode) + ', AE')
        print('accuracy_global: NoW = ', now, ', mode = ', mode, 'AE --> Final value = ',
              np.mean(data['accuracy_global'][-1, :]))
        # Obtain the baseline results
        save_file = 'results_baseline_' + str(now) + '_' + str(mode) + '.pickle'
        try:
            with open(save_file, 'rb') as handle:
                data = pickle.load(handle)
        except:
            data['accuracy_global'] = np.zeros((epochs, now))
        tab[1, n_of_workers.index(now), modes.index(mode)] = np.mean(data['accuracy_global'][-1, :])
        plt.plot(x_ep, 1 - np.mean(data['accuracy_global'], axis=1), color=color_dict[str(now)],
                 marker=mark_dict['baseline'], linestyle=line_dict[str(mode)],
                 label='NoW = ' + str(now) + ', mode = ' + str(mode) + ', baseline')
        print('accuracy_global: NoW = ', now, ', mode = ', mode, 'baseline --> Final value = ',
              np.mean(data['accuracy_global'][-1, :]))
    plt.title('Global acc')
    plt.xlabel('Epoch')
    plt.ylabel('1 - accuracy')
    # plt.legend(loc='best')
    tikz_save('acc_global.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
    plt.savefig('acc_global.png', bbox_inches='tight')

    for mode in modes:
        print('For ', mode)
        print(tabulate.tabulate(tab[:, :, modes.index(mode)], headers=n_of_workers, showindex=['ae', 'bas'],
                                tablefmt='latex'))
    # plt.show()
    plt.close()

