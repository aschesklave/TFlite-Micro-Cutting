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

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_modifier.h"

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  ModelModifier *modifier = nullptr;
  constexpr int kTensorArenaSize = 100000;
  uint8_t tensor_arena[kTensorArenaSize];
  uint32_t TARGET_LAYER = 0;
  uint32_t TARGET_SHAPE = 10;
}  // namespace

const float* images[10];

void initializeImages() {
  images[0] = x_test_0class;
  images[1] = x_test_1class;
  images[2] = x_test_2class;
  images[3] = x_test_3class;
  images[4] = x_test_4class;
  images[5] = x_test_5class;
  images[6] = x_test_6class;
  images[7] = x_test_7class;
  images[8] = x_test_8class;
  images[9] = x_test_9class;
}

uint32_t measureTime(tflite::MicroInterpreter* interpreter, int runs, tflite::ErrorReporter* error_reporter)
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
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed!");
      return 0;
    }
    float max_percentage = -1;
    unsigned int prediction = 666;
    for(int class_idx = 0; class_idx < 10; ++class_idx)
    {
      TF_LITE_REPORT_ERROR(error_reporter, "Class %d: %f", class_idx, output->data.f[class_idx]);
      Serial1.print("Class "); Serial1.print(class_idx); Serial1.print(": "); Serial1.println(output->data.f[class_idx]);
      if(output->data.f[class_idx] > max_percentage)
      {
        max_percentage = output->data.f[class_idx];
        prediction = class_idx;
      }
    }
    TF_LITE_REPORT_ERROR(error_reporter, "Max prob: %f; Prediction: class %d; Correct: %d", max_percentage, prediction, img_no);
    Serial1.print("Max prob: "); Serial1.print(max_percentage); Serial1.print("; Prediction: class ");
    Serial1.print(prediction); Serial1.print("; Correct: "); Serial1.println(img_no);
  }
  return micros() - start_time;
}

__attribute__((optimize(0))) void setup() {
  Serial1.begin(115200);
  while (!Serial1);
  Serial1.println("Starting...");
  tflite::InitializeTarget();

  initializeImages();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  {
    const tflite::Model* model_const = tflite::GetModel(custom_reds_tflite);
    model = const_cast<tflite::Model*>(model_const);
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static ModelModifier static_modifier(model, error_reporter);
  modifier = &static_modifier;

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial1.println("AllocateTensors() failed");
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  uint32_t duration = measureTime(interpreter, 100, error_reporter);
  Serial1.print("Duration: "); Serial1.println(duration);

  modifier->modifyFullyConnectedShape(TARGET_LAYER, TARGET_SHAPE);

  uint32_t modified_duration = measureTime(interpreter, 100, error_reporter);
  Serial1.print("Modified duration: "); Serial1.println(modified_duration);
}

void loop() { }
