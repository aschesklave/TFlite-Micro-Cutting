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
//#include "constants.h"
#include "digits_normalized.h"
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
  constexpr int kTensorArenaSize = 100000;
  uint8_t tensor_arena[kTensorArenaSize];
  uint32_t TARGET_SHAPE = 5;
}  // namespace

// const float* images[10];

// void initializeImages() {
//   images[0] = x_test_0class;
//   images[1] = x_test_1class;
//   images[2] = x_test_2class;
//   images[3] = x_test_3class;
//   images[4] = x_test_4class;
//   images[5] = x_test_5class;
//   images[6] = x_test_6class;
//   images[7] = x_test_7class;
//   images[8] = x_test_8class;
//   images[9] = x_test_9class;
// }

float measureTime(tflite::MicroInterpreter* interpreter)
{
  const int num_images = 1797;
  unsigned int num_correct = 0;
  uint32_t start_time = micros();
  for(int i = 0; i < num_images; ++i)
  {
    const float* curr_img = images[i];
    float* image_data = input->data.f;
    for(unsigned int i = 0; i < 8 * 8; ++i)
    {
      *image_data++ = curr_img[i];
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed!");
      return 0;
    }
    float max_percentage = -1;
    int prediction = 666;
    for(int class_idx = 0; class_idx < 10; ++class_idx)
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
  return (micros() - start_time) / (float)num_images;
}

__attribute__((optimize(0))) void setup() {
  Serial1.begin(115200);
  while (!Serial1);
  Serial1.println("Starting...");
  tflite::InitializeTarget();

  // initializeImages();

  {
    const tflite::Model* model_const = tflite::GetModel(second_model_activations_tflite);
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

  float duration = measureTime(interpreter);
  Serial1.print("Duration: "); Serial1.println(duration);

  modifier->modifyFullyConnectedShape(0, TARGET_SHAPE);
  modifier->modifyFullyConnectedShape(2, TARGET_SHAPE);
  modifier->modifyFullyConnectedShape(4, TARGET_SHAPE);
  modifier->modifyFullyConnectedShape(6, TARGET_SHAPE);
  modifier->modifyFullyConnectedShape(8, TARGET_SHAPE);

  float modified_duration = measureTime(interpreter);
  Serial1.print("Modified duration: "); Serial1.println(modified_duration);
}

void loop() { }
