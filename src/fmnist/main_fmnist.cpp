#include <Arduino.h>
#include <TensorFlowLite.h>

#include "model_fmnist.h"
#include "images.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 150000;
uint8_t tensor_arena[kTensorArenaSize];

const uint8_t* images[10];
unsigned int truth[10];

void setup() {
  delay(5000);
  pinMode(LED_BUILTIN, OUTPUT);
  tflite::InitializeTarget();
  digitalWrite(LED_BUILTIN, 0);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  TF_LITE_REPORT_ERROR(error_reporter, "Starting Setup proc");
  model = tflite::GetModel(python_model_tflite);

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  output = interpreter->output(0);
  input = interpreter->input(0);
  if ((input->dims->size != 4) || (input->dims->data[0] != 1) ||
      (input->dims->data[1] != (int)size) ||
      (input->dims->data[2] != (int)size) ||
      (input->dims->data[3] != 1) || (input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Dims: %d | %d - %d - %d - %d", input->dims->size, input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
  inference_count = 0;
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

  truth[0] = y_0;
  truth[1] = y_1;
  truth[2] = y_2;
  truth[3] = y_3;
  truth[4] = y_4;
  truth[5] = y_5;
  truth[6] = y_6;
  truth[7] = y_7;
  truth[8] = y_8;
  truth[9] = y_9;
  
  TF_LITE_REPORT_ERROR(error_reporter, "Starting loop ...");
}

void loop() {
  digitalWrite(LED_BUILTIN, 1);
  TF_LITE_REPORT_ERROR(error_reporter, "Running Inference!");
  for(int img_no = 0; img_no < 10; ++img_no)
  {
    //TF_LITE_REPORT_ERROR(error_reporter, "Loading data for image %d", img_no);
    const uint8_t* curr_img = images[img_no];
    float* image_data = input->data.f;
    for(unsigned int i = 0; i < size * size; ++i)
    {
      *image_data++ = curr_img[i] / 255.0f;
    }
    //TF_LITE_REPORT_ERROR(error_reporter, "Running inference ...");
    if (kTfLiteOk != interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    }
    TfLiteTensor* output = interpreter->output(0);
    float max_percentage = -1;
    unsigned int prediction = 666;
    for(int class_idx = 0; class_idx < 10; ++class_idx)
    {
      //TF_LITE_REPORT_ERROR(error_reporter, "Class %d: %f", class_idx, output->data.f[class_idx]);
      if(output->data.f[class_idx] > max_percentage)
      {
        max_percentage = output->data.f[class_idx];
        prediction = class_idx;
      }
    }
    if(prediction == truth[img_no])
    {
      TF_LITE_REPORT_ERROR(error_reporter, "Prediction for image %d was correct!", img_no);
    }
    else
    {
      TF_LITE_REPORT_ERROR(error_reporter, "Prediction for image %d was false! Truth: %d | Prediction: %d", img_no, truth[img_no], prediction);
    }
  }
  digitalWrite(LED_BUILTIN, 0);
  delay(10000);
}