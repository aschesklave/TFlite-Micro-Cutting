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
  uint32_t last_read = 0u;
  constexpr uint32_t kWaitTime = 500u;
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

  input = interpreter->input(0);
  output = interpreter->output(0);

  constexpr uint32_t new_shape = 14;

  modifier->modify2DConvolutionalShape(0, new_shape);
  modifier->modify2DConvolutionalShape(2, new_shape);
  modifier->modify2DConvolutionalShape(4, new_shape);
}

void printSerialized(arduino::UART* interface, uint8_t prediction, uint32_t time)
{
  interface->print("{\"prediction\":\"");
  interface->print(prediction);
  interface->print("\",\"time\":\"");
  interface->print(time);
  interface->println("\"}");
}

void loop() {
  /* Reset last read time if there is no data in memory. */
  if (0u == read_index) {
    last_read = millis();
  }
  /* Look for incoming transmission. */
  if (Serial1.available() > 0) {
    /* If the last incoming transmission was too long ago reset read index before writing to buffer. */
    if (millis() - last_read > kWaitTime) {
      read_index = 0u;
    }
    /* Write current incoming byte to buffer. */
    input->data.raw[read_index++] = static_cast<char>(Serial1.read());
    /* Update last read index. */
    last_read = millis();
  }

  /* If a complete sample was received. */
  if (byte_size == read_index) {
    read_index = 0u;
    TfLiteTensor *output = interpreter->output(0);
    /* Save the start time, */
    uint32_t start_time = micros();
    /* Invoke the inference. */
    TfLiteStatus invoke_status = interpreter->Invoke();
    /* Calculate inference time. */
    uint32_t inference_time = micros() - start_time;
    /* If there was an error during the inference cancel and print error. */
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed!\r\n");
    }
    /* If the inference was successful. */
    else
    {
      /* Get index of prediction. */
      float max_percentage = -1;
      uint8_t prediction = 255;
      for (uint8_t class_idx = 0; class_idx < 8; ++class_idx)
      {
        if (output->data.f[class_idx] > max_percentage)
        {
          max_percentage = output->data.f[class_idx];
          prediction = class_idx;
        }
      }
      /* Send prediction back to Host. */
      printSerialized(&Serial1, prediction, inference_time);
    }
  }
}
