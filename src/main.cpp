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

#include "custom_stack_allocator.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 100000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

const float* images[10];

class bypass_private;

template <>
bypass_private* tflite::MicroInterpreter::typed_input_tensor<bypass_private>(int tensor_index) {
  model_ = reinterpret_cast<Model*>(tensor_index);
  return nullptr;
}

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

uint32_t measure_time(tflite::MicroInterpreter* interpreter, int runs, tflite::ErrorReporter* error_reporter)
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

__attribute__((optimize(0))) void modify_model(tflite::MicroInterpreter* interpreter, tflite::Model* unmodified_model)
{
  auto unpacked_model = unmodified_model->UnPack();
  auto& subgraphs = unpacked_model->subgraphs;
  auto& tensors = subgraphs[0]->tensors;

  int target_layer = 3;
  auto& shape = tensors[target_layer]->shape;
  shape[0] = 12;

  static char inst_memory[sizeof(flatbuffers::FlatBufferBuilder)];
  flatbuffers::FlatBufferBuilder* fbb =
      new (inst_memory) flatbuffers::FlatBufferBuilder(
          90000,
          &CustomStackAllocator::instance(16));

  Serial1.println("before pack");
  TF_LITE_REPORT_ERROR(error_reporter, "before pack");

  auto model_offset = tflite::Model::Pack(*fbb, unpacked_model);

  Serial1.println("after pack");
  TF_LITE_REPORT_ERROR(error_reporter, "after pack");

  tflite::FinishModelBuffer(*fbb, model_offset);
  void* model_pointer = fbb->GetBufferPointer();
  const tflite::Model* tmp_model = flatbuffers::GetRoot<tflite::Model>(model_pointer);
  tflite::Model* new_model = const_cast<tflite::Model*>(tmp_model);
  interpreter->typed_input_tensor<bypass_private>((int)new_model);
  delete fbb;
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

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  uint32_t duration = measure_time(interpreter, 10, error_reporter);

  modify_model(interpreter, model);

  uint32_t modified_duration = measure_time(interpreter, 10, error_reporter);

  Serial1.print("Duration unmodified: ");
  Serial1.println(duration);

  Serial1.print("Duration modified: ");
  Serial1.println(modified_duration);
}

__attribute__((optimize(0))) void loop() {}
