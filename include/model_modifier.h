#pragma once

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "custom_stack_allocator.h"

class ModelModifier
{
  private:
    uint8_t findOpCodeIndex(const tflite::BuiltinOperator op, uint32_t &index);

    uint32_t op_index_fully_connected_;
    tflite::MicroInterpreter* interpreter_;
    tflite::ModelT* unpacked_model_;
    tflite::SubGraphT* subgraph_;
    std::vector<std::unique_ptr<tflite::OperatorCodeT>>* opcodes_;
    tflite::ErrorReporter* error_reporter_;

  public:
    void modifyShape(const int32_t layer_index, const int32_t new_shape);
    int32_t getWeightTensorIndex(const int32_t& target_op_index);
    uint8_t setTensorShape(const int32_t tensor_index, const int32_t new_shape);

    ModelModifier(tflite::MicroInterpreter* interpreter, tflite::Model* unmodified_model, tflite::ErrorReporter* error_reporter);
    ~ModelModifier();
};

