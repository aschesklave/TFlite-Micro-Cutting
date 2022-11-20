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

uint32_t TARGET_LAYER = 1;
uint32_t TARGET_SHAPE = 8;

static uint32_t measureTime(tflite::MicroInterpreter* interpreter, int runs)
{
  TfLiteTensor* input = nullptr;
  float x = 0.5;
  input->data.f[0] = x;
  uint32_t start_time = micros();
  for(int i = 0; i < runs; ++i)
  {
    TfLiteStatus invoke_status = interpreter->Invoke();
  }
  return micros() - start_time;
}

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
ModelModifier *modifier = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

__attribute__((optimize(0))) void setup() {
  tflite::InitializeTarget();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  {
    const tflite::Model* model_const = tflite::GetModel(custom_model_float_tflite);
    model = const_cast<tflite::Model*>(model_const);
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  //uint32_t duration = measureTime(interpreter, 100);

  modifier = new ModelModifier(interpreter, model, error_reporter);
  modifier->modifyShape(TARGET_LAYER, TARGET_SHAPE);

  uint32_t modified_duration = measureTime(interpreter, 100);

  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;
}

__attribute__((optimize(0))) void loop() {
  //float position = static_cast<float>(inference_count) /
  //                 static_cast<float>(kInferencesPerCycle);
  //float x = position * kXrange;
  float x = 0.5;

  //int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  //input->data.int8[0] = x_quantized;
  input->data.f[0] = x;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
    return;
  }

  //int8_t y_quantized = output->data.int8[0];
  //float y = (y_quantized - output->params.zero_point) * output->params.scale;
  float y = output->data.f[0];

  HandleOutput(error_reporter, x, y);

  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
