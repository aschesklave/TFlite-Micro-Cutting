import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_acc_time():
    data_list = []
    # data_dict = {'model_1_acc': [86.03, 79.14, 66.03, 39.01], 'model_1_time': [614264.12, 215747.34, 85451.85, 31686.02],
    #              'point_labels': ['100%', '50%', '25%', '12.5%'],
    #              'filename': 'cnn'
    #              }
    # data_list.append(data_dict)

    # 400% data
    data_dict = {'model_1_acc': [98.89, 98.66, 98.22, 92.93, 44.46, 18.25], 'model_1_time': [617.92, 365.16, 238.87, 175.69, 137.15, 124.29],
                 'model_2_acc': [98.33, 97.83, 97.44, 24.76, 9.91, 9.91], 'model_2_time': [3293.64, 1854.44, 1135.83, 773.42, 556.07, 484.25],
                 'model_3_acc': [98.33, 98.33, 98.44, 95.72, 64.83, 23.43], 'model_3_time': [1002.22, 618.38, 426.57, 330.60, 272.81, 253.39],
                 'point_labels': ['400%', '200%', '100%', '50%', '25%', '12.5%'],
                 'filename': 'mnist_400'
                 }
    data_list.append(data_dict)

    # data_dict = {'model_1_acc': [99.44, 96.22, 56.71, 26.38], 'model_1_time': [239.51, 176.42, 137.72, 124.71],
    #              'model_2_acc': [98.66, 85.59, 10.13, 10.13], 'model_2_time': [581.16, 441.99, 352.39, 324.56],
    #              'model_3_acc': [99.83, 96.88, 66.50, 36.78], 'model_3_time': [427.33, 331.46, 273.64, 253.96],
    #              'point_labels': ['100%', '50%', '25%', '12.5%'],
    #              'filename': 'mnist'
    #              }
    # data_list.append(data_dict)

    # data_dict = {'model_1_acc': [99.30, 99.30, 98.95, 98.59], 'model_1_time': [125.31, 96.63, 79.39, 73.71],
    #              'model_2_acc': [99.82, 99.65, 62.74, 62.74], 'model_2_time': [455.87, 345.44, 278.95, 256.77],
    #              'model_3_acc': [99.47, 99.47, 99.47, 99.12], 'model_3_time': [261.20, 193.08, 152.19, 138.58],
    #              'point_labels': ['100%', '50%', '25%', '12.5%'],
    #              'filename': 'cancer'
    #              }
    # data_list.append(data_dict)
    #
    # data_dict = {'model_1_acc': [100.0, 100.0, 99.44, 60.11], 'model_1_time': [105.94, 90.33, 80.34, 77.32],
    #              'model_2_acc': [99.44, 99.44, 97.75, 60.11], 'model_2_time': [443.44, 346.73, 288.75, 269.19],
    #              'model_3_acc': [100.0, 100.0, 82.58, 82.58], 'model_3_time': [247.69, 193.60, 160.62, 149.74],
    #              'point_labels': ['100%', '50%', '25%', '12.5%'],
    #              'filename': 'wine'
    #              }
    # data_list.append(data_dict)

    markers = ['o', 's', 'D', '^']
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='100%', markerfacecolor='k', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='50%', markerfacecolor='k', markersize=8),
        Line2D([0], [0], marker='D', color='w', label='25%', markerfacecolor='k', markersize=8),
        Line2D([0], [0], marker='^', color='w', label='12.5%', markerfacecolor='k', markersize=8),
        Line2D([0], [0], color='b', label='1st Model', linestyle='--'),
        Line2D([0], [0], color='r', label='2nd Model', linestyle='--'),
        Line2D([0], [0], color='g', label='3rd Model', linestyle='--'),
    ]

    for d in data_list:
        fig, ax = plt.subplots()

        plt.xlabel('Inference time [μs]')
        plt.ylabel('Accuracy [%]')
        ax.margins(x=0.1, y=0.1)
        # ax.set_ylim([0, 110])

        ax.plot(d['model_1_time'], d['model_1_acc'], linewidth=1.0, linestyle='--', marker='', color='b',
                label='First Model')
        ax.plot(d['model_2_time'], d['model_2_acc'], linewidth=1.0, linestyle='--', marker='', color='r',
                label='Second Model')
        ax.plot(d['model_3_time'], d['model_3_acc'], linewidth=1.0, linestyle='--', marker='', color='g',
                label='Third Model')

        for i, j, m in zip(d['model_1_time'], d['model_1_acc'], markers):
            plt.scatter(i, j, marker=m, color='b')
        for i, j, m in zip(d['model_2_time'], d['model_2_acc'], markers):
            plt.scatter(i, j, marker=m, color='r')
        for i, j, m in zip(d['model_3_time'], d['model_3_acc'], markers):
            plt.scatter(i, j, marker=m, color='g')

        ax.legend(handles=legend_elements, loc='lower right')
        plt.savefig(f'plots/{d["filename"]}_acc_time.pdf', format='pdf', bbox_inches='tight')
        plt.show()

def plot_size_time():
    data_list = []
    # data_dict = {'model_1_time': [239.51, 176.42, 137.72, 124.71],
    #              'filename': 'cnn'
    #              }
    # data_list.append(data_dict)

    # 400%
    data_dict = {'model_1_time': [617.92, 365.16, 238.87, 175.69, 137.15, 124.29],
                 'model_2_time': [3293.64, 1854.44, 1135.83, 773.42, 556.07, 484.25],
                 'model_3_time': [1002.22, 618.38, 426.57, 330.60, 272.81, 253.39],
                 'filename': 'mnist_400'
                 }
    data_list.append(data_dict)

    # data_dict = {'model_1_time': [239.51, 176.42, 137.72, 124.71],
    #              'model_2_time': [581.16, 441.99, 352.39, 324.56],
    #              'model_3_time': [427.33, 331.46, 273.64, 253.96],
    #              'filename': 'mnist'
    #              }
    # data_list.append(data_dict)

    # data_dict = {'model_1_time': [125.31, 96.63, 79.39, 73.71],
    #              'model_2_time': [455.87, 345.44, 278.95, 256.77],
    #              'model_3_time': [261.20, 193.08, 152.19, 138.58],
    #              'filename': 'cancer'
    #              }
    # data_list.append(data_dict)
    #
    # data_dict = {'model_1_time': [105.94, 90.33, 80.34, 77.32],
    #              'model_2_time': [443.44, 346.73, 288.75, 269.19],
    #              'model_3_time': [247.69, 193.60, 160.62, 149.74],
    #              'filename': 'wine'
    #              }
    # data_list.append(data_dict)

    sizes = [400, 200, 100, 50, 25, 12.5]
    for d in data_list:
        fig, ax = plt.subplots()
        plt.xlabel('Model size [%]')
        plt.ylabel('Inference time [μs]')
        ax.plot(sizes, d['model_1_time'], linewidth=1.0, marker='', color='b',
                label='First Model')
        ax.plot(sizes, d['model_2_time'], linewidth=1.0, marker='', color='r',
                label='Second Model')
        ax.plot(sizes, d['model_3_time'], linewidth=1.0, marker='', color='g',
                label='Third Model')
        ax.legend(loc='upper left')
        plt.savefig(f'plots/{d["filename"]}_size_time.pdf', format='pdf', bbox_inches='tight')
        plt.show()


plot_size_time()
plot_acc_time()