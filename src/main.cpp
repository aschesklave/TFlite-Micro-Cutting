/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>
#include <Arduino.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "samples.h"
// #include "digits_normalized.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_modifier.h"

// python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html

namespace {
  tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  ModelModifier *modifier = nullptr;
  constexpr int kTensorArenaSize = 169000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr uint16_t byte_size = 1280 * 4;
  uint16_t read_index = 0u;
}  // namespace

float measureTimeConv(tflite::MicroInterpreter* interpreter, int runs) {
  int correct = 0;
  int num_samples = 20;
  const int total_predictions = num_samples * runs;

  uint32_t start_time = micros();
  for(int i = 0; i < runs; ++i) {
    for(int sample_no = 0; sample_no < num_samples; ++sample_no) {
      memcpy(input->data.raw, &samples[sample_no][0], size);
      TfLiteTensor* output = interpreter->output(0);
      float max_percentage = -1;
      int prediction = 666;
      for(int class_idx = 0; class_idx < 8; ++class_idx) {
        if(output->data.f[class_idx] > max_percentage) {
          max_percentage = output->data.f[class_idx];
          prediction = class_idx;
        }
      }
      if(prediction == labels[sample_no]) {
        correct++;
        // MicroPrintf("Prediction for sample %d was correct!", sample_no);
        // Serial1.print("Prediction for sample ");
        // Serial1.print(sample_no);
        // Serial1.println(" was correct!");
      }
      else {
        // MicroPrintf("Prediction for sample %d was false! Truth: %d | Prediction: %d", sample_no, labels[sample_no], prediction);
        // Serial1.print("Prediction for sample ");
        // Serial1.print(sample_no);
        // Serial1.print(" was false! Truth: ");
        // Serial1.print(labels[sample_no]);
        // Serial1.print(" | Prediction: ");
        // Serial1.println(prediction);
      }
    }
  }
  Serial1.print("Accuracy: ");
  Serial1.println((float)correct / total_predictions);
  return micros() - start_time;
}

float measureTimeFC(tflite::MicroInterpreter* interpreter)
{
  const int num_images = 1797;
  const int num_classes = 10;

  unsigned int num_correct = 0;
  uint32_t start_time = micros();
  for(int i = 0; i < num_images; ++i)
  {
    const float* curr_img = samples[i];
    float* sample_data = input->data.f;
    for(unsigned int i = 0; i < size; ++i)
    {
      *sample_data++ = curr_img[i];
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed!");
      return 0;
    }

    float max_percentage = -1;
    int prediction = 666;
    for(int class_idx = 0; class_idx < num_classes; ++class_idx)
    {
      //MicroPrintf("Class %d: %f", class_idx, output->data.f[class_idx]);
      //Serial1.print("Class "); Serial1.print(class_idx); Serial1.print(": "); Serial1.println(output->data.f[class_idx]);
      if(output->data.f[class_idx] > max_percentage)
      {
        max_percentage = output->data.f[class_idx];
        prediction = class_idx;
      }
    }
    int truth = labels[i];
    if(truth == prediction) {
      num_correct += 1;
    }
    //MicroPrintf("Max prob: %f; Prediction: class %d; Correct: %d", max_percentage, prediction, truth);
    //Serial1.print("Max prob: "); Serial1.print(max_percentage); Serial1.print("; Prediction: class ");
    //Serial1.print(prediction); Serial1.print("; Correct: "); Serial1.println(truth);
  }
  float acc = num_correct / (float)num_images;
  Serial1.print("Accuracy: ");Serial1.println(acc * 100);
  Serial.print("Accuracy: ");Serial.println(acc * 100);
  return (micros() - start_time) / (float)num_images;
}

void setup() {
  Serial1.begin(115200);
  while (!Serial1);
  //Serial1.println("Starting...");
  tflite::InitializeTarget();

  {
    const tflite::Model* model_const = tflite::GetModel(REDS_cnn_three_convolutions_tflite);
    model = const_cast<tflite::Model*>(model_const);
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static ModelModifier static_modifier(model);
  modifier = &static_modifier;

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial1.println("AllocateTensors() failed");
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  if (Serial1.available() > 0) input->data.raw[read_index++] = static_cast<char>(Serial1.read());

  if (byte_size == read_index) {
    read_index = 0u;
    Serial.println();
    TfLiteTensor *output = interpreter->output(0);
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed!");
    }
    else
    {
      float max_percentage = -1;
      int prediction = 666;
      for (int class_idx = 0; class_idx < 8; ++class_idx)
      {
        if (output->data.f[class_idx] > max_percentage)
        {
          max_percentage = output->data.f[class_idx];
          prediction = class_idx;
        }
      }
      Serial1.println(prediction);
    }
  }
}
