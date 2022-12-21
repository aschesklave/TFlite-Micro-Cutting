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

#include "images.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_modifier.h"

namespace {
  tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  ModelModifier *modifier = nullptr;
  constexpr int kTensorArenaSize = 100000;
  uint8_t tensor_arena[kTensorArenaSize];
  uint32_t TARGET_LAYER = 0;
  uint32_t TARGET_SHAPE = 11;
}  // namespace

const float* images[10];
int labels[10];

void initializeImages() {
  images[0] = img_0;
  images[1] = img_1;
  images[2] = img_2;
  images[3] = img_3;
  images[4] = img_4;
  images[5] = img_5;
  images[6] = img_6;
  images[7] = img_7;
  images[8] = img_8;
  images[9] = img_9;
}

void initializeLabels() {
  labels[0] = y_0;
  labels[1] = y_1;
  labels[2] = y_2;
  labels[3] = y_3;
  labels[4] = y_4;
  labels[5] = y_5;
  labels[6] = y_6;
  labels[7] = y_7;
  labels[8] = y_8;
  labels[9] = y_9;
}

uint32_t measureTime(tflite::MicroInterpreter* interpreter, int runs)
{
  uint32_t start_time = micros();
  for(int i = 0; i < runs; ++i)
  {
    int img_no = i % 10;
    const float* curr_img = images[img_no];
    float* image_data = input->data.f;
    for(unsigned int i = 0; i < 32 * 32; ++i)
    {
      *image_data++ = curr_img[i];
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed!");
      return 0;
    }
    float max_percentage = -1;
    unsigned int prediction = 666;
    for(int class_idx = 0; class_idx < 10; ++class_idx)
    {
      // MicroPrintf("Class %d: %f", class_idx, output->data.f[class_idx]);
      // Serial1.print("Class "); Serial1.print(class_idx); Serial1.print(": "); Serial1.println(output->data.f[class_idx]);
      if(output->data.f[class_idx] > max_percentage)
      {
        max_percentage = output->data.f[class_idx];
        prediction = class_idx;
      }
    }
    // MicroPrintf("Max prob: %f; Prediction: class %d; Correct: %d", max_percentage, prediction, labels[img_no]);
    // Serial1.print("Max prob: "); Serial1.print(max_percentage); Serial1.print("; Prediction: class ");
    // Serial1.print(prediction); Serial1.print("; Correct: "); Serial1.println(labels[img_no]);
  }
  return micros() - start_time;
}

__attribute__((optimize(0))) void setup() {
  Serial1.begin(115200);
  while (!Serial1);
  Serial1.println("Starting...");
  tflite::InitializeTarget();

  initializeImages();
  initializeLabels();

  {
    const tflite::Model* model_const = tflite::GetModel(custom_reds_tflite);
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

  uint32_t duration = measureTime(interpreter, 100);
  Serial1.print("Duration: "); Serial1.println(duration);

  modifier->modifyFullyConnectedShape(TARGET_LAYER, TARGET_SHAPE);

  uint32_t modified_duration = measureTime(interpreter, 100);
  Serial1.print("Modified duration: "); Serial1.println(modified_duration);
}

void loop() { }
