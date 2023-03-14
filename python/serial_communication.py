import serial
import numpy as np
import json

test_data = np.loadtxt('data/wake_word_val_set_normalized.csv', delimiter=',', skiprows=1)

dataset_len = test_data.shape[0]
num_features = test_data.shape[1]
label_col = num_features - 1
it = 0
correct = 0
predictions = []
times = []
truths = test_data[:, label_col].astype(int)

ser = serial.Serial('COM3', 115200, timeout=5)

max_iter = 100
for i in range(max_iter):
    for row in test_data:
        row_f32 = row[:-1].astype('float32')
        sample = row_f32.tobytes()
        ser.reset_input_buffer()
        num_b = ser.write(sample)
        mc_resp = ser.readline()
        if not mc_resp:
            print('No response from MC, retrying ...')
            break

        resp_dict = json.loads(mc_resp)
        pred = int(resp_dict['prediction'])
        truth = truths[it]
        if pred == truth:
            correct += 1
        predictions.append(pred)
        times.append(float(resp_dict['time']))
        it += 1
        running_accuracy = correct / it
        print(f'{it}/{dataset_len}> Pred: {int(resp_dict["prediction"])} ({truths[it-1]}) | '
              f'Running acc: {running_accuracy} | Infer time: {float(resp_dict["time"])}')

    if it == dataset_len:
        break

predictions = np.array(predictions)
comp = np.equal(truths, predictions)
correct = np.count_nonzero(comp)
accuracy = correct / dataset_len

infer_times = np.array(times)
mean_infer_time = infer_times.sum() / dataset_len

print(f'Accuracy: {accuracy}')
print(f'Mean inference time: {mean_infer_time}')