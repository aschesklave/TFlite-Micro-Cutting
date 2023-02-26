import serial
import numpy as np
import json

test_data = np.loadtxt('data/wake_word_val_set_normalized.csv', delimiter=',', skiprows=1)

dataset_len = test_data.shape[0]
num_features = test_data.shape[1]
label_col = num_features - 1
it = 0
predictions = []
times = []
truths = test_data[:, label_col].astype(int)

ser = serial.Serial('COM3', 115200)
for row in test_data:
    row_f32 = row[:-1].astype('float32')
    sample = row_f32.tobytes()
    ser.reset_input_buffer()
    num_b = ser.write(sample)
    resp_dict = json.loads(ser.readline())
    predictions.append(int(resp_dict['prediction']))
    times.append(float(resp_dict['time']))
    it += 1
    print(f'{it}/{dataset_len} samples predicted | Pred: {int(resp_dict["prediction"])} ({truths[it-1]}) | Infer time: {float(resp_dict["time"])}')

predictions = np.array(predictions)
comp = np.equal(truths, predictions)
correct = np.count_nonzero(comp)
accuracy = correct / dataset_len

infer_times = np.array(times)
mean_infer_time = infer_times.sum() / dataset_len

print(f'Accuracy: {accuracy}')
print(f'Mean inference time: {mean_infer_time}')