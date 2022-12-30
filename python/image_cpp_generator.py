import pandas as pd
csv_filename = 'digits_normalized.csv'
cpp_filename = 'digits_normalized.cpp'

#image_df = pd.read_csv('digits_normalized.csv')

img_string = ''
label_string = ''

with open(csv_filename, 'r') as f:
    for line in f:
        if line[-2] == 'l':
            continue
        image = f'{{{line[:-3]}}},\n'
        label = f'{line[-2]},'
        img_string += image
        label_string += label

img_init_string = f'const float images[1797][64] = {{{img_string[:-2]}}};'
label_init_string = f'const int labels[1797] = {{{label_string[:-1]}}};'
cpp_file = f'#include "digits_normalized.h"\n\n{img_init_string}\n\n{label_init_string}'

with open(cpp_filename, 'w') as f:
    f.write(cpp_file)
