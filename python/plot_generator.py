import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data_list = []
data_dict = {'model_1_acc': [99.44, 96.22, 56.71, 26.38], 'model_1_time': [352.90, 241.08, 172.57, 149.46],
             'model_2_acc': [98.66, 85.59, 10.13, 10.13], 'model_2_time': [833.41, 591.16, 437.90, 389.55],
             'model_3_acc': [99.83, 96.88, 66.50, 36.78], 'model_3_time': [666.47, 495.31, 393.26, 357.59],
             'point_labels': ['100%', '50%', '25%', '12.5%'],
             'filename': 'mnist'
             }
data_list.append(data_dict)

data_dict = {'model_1_acc': [99.30, 99.30, 98.95, 98.59], 'model_1_time': [172.84, 123.67, 94.17, 84.34],
             'model_2_acc': [99.82, 99.65, 62.74, 62.74], 'model_2_time': [643.66, 455.74, 342.54, 304.98],
             'model_3_acc': [99.47, 99.47, 99.47, 99.12], 'model_3_time': [387.68, 296.00, 197.89, 174.21],
             'point_labels': ['100%', '50%', '25%', '12.5%'],
             'filename': 'cancer'
             }
data_list.append(data_dict)

data_dict = {'model_1_acc': [100.0, 100.0, 99.44, 60.11], 'model_1_time': [136.20, 110.02, 93.92, 88.67],
             'model_2_acc': [99.44, 99.44, 97.75, 60.11], 'model_2_time': [605.15, 442.04, 343.89, 311.15],
             'model_3_acc': [100.0, 100.0, 82.58, 82.58], 'model_3_time': [362.34, 268.26, 211.03, 192.44],
             'point_labels': ['100%', '50%', '25%', '12.5%'],
             'filename': 'wine'
             }
data_list.append(data_dict)

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
    plt.show()
    plt.savefig(f'plots/{d["filename"]}.eps', format='eps', bbox_inches='tight')
    plt.close()