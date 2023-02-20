csv_filename = 'data/wake_word_val_set_normalized.csv'
cpp_filename = 'data/wake_words.cpp'

img_string = ''
label_string = ''
num_samples = 20
num_features = 1280
with open(csv_filename, 'r') as f:
    i = 0
    for line in f:
        i += 1
        if i == 1:
            continue
        elif i == num_samples + 2:
            break

        if line[-3] == '.':
            line_offset = 2
        else:
            line_offset = 0

        image = f'{{{line[:-3 - line_offset]}}},\n'
        label = f'{line[-2 - line_offset]},'
        img_string += image
        label_string += label


img_init_string = f'const float samples[{num_samples}][{num_features}] = {{{img_string[:-2]}}};'
label_init_string = f'const int labels[{num_samples}] = {{{label_string[:-1]}}};'
cpp_file = f'#include "samples.h"\n\nconst unsigned int size = {num_features};\n\n{img_init_string}\n\n{label_init_string}'

with open(cpp_filename, 'w') as f:
    f.write(cpp_file)
