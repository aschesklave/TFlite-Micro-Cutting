import pandas as pd
csv_filename = 'data/mnist_val_normalized_truncated.csv'
cpp_filename = 'data/images.cpp'

img_string = ''
label_string = ''

with open(csv_filename, 'r') as f:
    for line in f:
        if line[-2] == 'l':
            continue

        if line[-3] == '.':
            line_offset = 2
        else:
            line_offset = 0

        image = f'{{{line[:-3 - line_offset]}}},\n'
        label = f'{line[-2 - line_offset]},'
        img_string += image
        label_string += label

img_init_string = f'const float images[1797][64] = {{{img_string[:-2]}}};'
label_init_string = f'const int labels[1797] = {{{label_string[:-1]}}};'
cpp_file = f'#include "images.h"\n\n{img_init_string}\n\n{label_init_string}'

with open(cpp_filename, 'w') as f:
    f.write(cpp_file)
