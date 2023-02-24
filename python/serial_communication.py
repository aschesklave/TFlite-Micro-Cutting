import serial
import numpy as np

test_data = np.loadtxt('data/wake_word_val_set_normalized.csv', delimiter=',', skiprows=1)

total = len(test_data)
it = 0
predictions = []
truths = test_data[:, 1280].astype(int)

ser = serial.Serial('COM3', 115200)
for row in test_data:
    row_f32 = row[:-1].astype('float32')
    sample = row_f32.tobytes()
    ser.reset_input_buffer()
    num_b = ser.write(sample)
    pred = int(ser.readline().decode('ascii')[0])
    predictions.append(pred)
    it += 1
    print(f'{it}/{total} samples predicted')

predictions = np.array(predictions)
comp = np.equal(truths, predictions)
correct = np.count_nonzero(comp)
print(f'Accuracy: {correct / total}')