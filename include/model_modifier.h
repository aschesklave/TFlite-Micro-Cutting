#pragma once

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

class ModelModifier
{
  private:
    uint8_t findOpCodeIndex(const tflite::BuiltinOperator op, int32_t &index);

    int32_t op_index_fully_connected_;
    int32_t op_index_2d_convolutional_;
    int32_t op_index_reshape_;
    tflite::Model* model_;
    tflite::ErrorReporter* error_reporter_;

  public:
    void modifyFullyConnectedShape(const int32_t layer_index, const int32_t new_shape);
    void modify2DConvolutionalShape(const int32_t layer_index, const int32_t new_shape);
    int32_t getMultipliedTensorShape(const int32_t tensor_index);
    int32_t getInputTensorIndex(const uint32_t& target_op_index);
    int32_t getOutputTensorIndex(const uint32_t& target_op_index);
    int32_t getWeightTensorIndex(const uint32_t& target_op_index);
    int8_t setTensorShape(const int32_t tensor_index, const int32_t new_shape, const uint32_t shape_index, int32_t &shape_diff);

    ModelModifier(tflite::Model* model);
    ~ModelModifier();
};

