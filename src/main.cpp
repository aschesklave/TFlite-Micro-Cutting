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

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace
class bypass_private;

template <>
bypass_private* tflite::MicroInterpreter::typed_input_tensor<bypass_private>(int tensor_index) {
  model_ = reinterpret_cast<Model*>(tensor_index);
  return nullptr;
}
// The name of this function is important for Arduino compatibility.
__attribute__((optimize(0))) void setup() {
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
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

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  
  auto unpacked_model = model->UnPack();
  auto& subgraphs = unpacked_model->subgraphs;
  auto& tensors = subgraphs[0]->tensors;

  int target_layer = 5;
  auto& shape = tensors[target_layer]->shape;
  shape[0] = 8;
  
  static char inst_memory[sizeof(flatbuffers::FlatBufferBuilder)];
  flatbuffers::FlatBufferBuilder* fbb =
      new (inst_memory) flatbuffers::FlatBufferBuilder(
          8192,
          &CustomStackAllocator::instance(16));
          
  auto model_offset = tflite::Model::Pack(*fbb, unpacked_model);

  tflite::FinishModelBuffer(*fbb, model_offset);
  void* model_pointer = fbb->GetBufferPointer();
  const tflite::Model* tmp_model = flatbuffers::GetRoot<tflite::Model>(model_pointer);
  tflite::Model* new_model = const_cast<tflite::Model*>(tmp_model);
  interpreter->typed_input_tensor<bypass_private>((int)new_model);

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
__attribute__((optimize(0))) void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  //float position = static_cast<float>(inference_count) /
  //                 static_cast<float>(kInferencesPerCycle);
  //float x = position * kXrange;
  float x = 0.5;

  // Quantize the input from floating-point to integer
  //int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // Place the quantized input in the model's input tensor
  //input->data.int8[0] = x_quantized;
  input->data.f[0] = x;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  //int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  //float y = (y_quantized - output->params.zero_point) * output->params.scale;
  float y = output->data.f[0];

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
