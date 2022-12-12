#include <Arduino.h>
#include "model_modifier.h"
#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"

ModelModifier::ModelModifier(tflite::Model* model, tflite::ErrorReporter* error_reporter)
  : model_(model), error_reporter_(error_reporter)
{
  if(findOpCodeIndex(tflite::BuiltinOperator_FULLY_CONNECTED, op_index_fully_connected_)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: No FULLY_CONNECTED layer found.");
    Serial1.println("ERROR: No FULLY_CONNECTED layer found.");
  }
}

ModelModifier::~ModelModifier() { }

uint8_t ModelModifier::findOpCodeIndex(const tflite::BuiltinOperator op, uint32_t& index) {
  const auto& opcode_vector = model_->operator_codes();
  for(auto it = opcode_vector->begin(); it != opcode_vector->end(); ++it) {
    if(it->builtin_code() == op) {
      index = std::distance(opcode_vector->begin(), it);
      return 0;
    }
  }
  return 1;
}

int32_t ModelModifier::getWeightTensorIndex(const int32_t& target_op_index) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if (target_op_index >= subgraph->operators()->size()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: layer index out of bounds.");
    Serial1.println("ERROR: layer index out of bounds.");
    return -1;
  }
  const tflite::Operator* target_op = (*subgraph->operators())[target_op_index];
  if (subgraph->operators()->size() < 3) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: selected layer needs 3 inputs but has %d", subgraph->operators()->size());
    Serial1.println("ERROR: selected layer needs 3 inputs but has "); Serial1.println(subgraph->operators()->size());
    return -1;
  }
  return (*target_op->inputs())[1];
}

uint8_t ModelModifier::setTensorShape(const int32_t tensor_index, const int32_t new_shape, const int32_t shape_index) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if (tensor_index >= subgraph->tensors()->size()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: tensor index out of bounds.");
    Serial1.println("ERROR: tensor index out of bounds.");
    return 1;
  }
  const tflite::Tensor* tensor = (*subgraph->tensors())[tensor_index];
  const flatbuffers::Vector<int32_t>* shape_vector_const = tensor->shape();
  if (shape_index >= shape_vector_const->size()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: shape index out of bounds.");
    Serial1.println("ERROR: shape index out of bounds.");
    return 1;
  }

  if(new_shape != (*shape_vector_const)[shape_index]) {
    if(new_shape > (*shape_vector_const)[shape_index]) {
      TF_LITE_REPORT_ERROR(error_reporter_, "WARNING: Increased shape.");
      Serial1.println("WARNING: Increased shape.");
    }
    flatbuffers::Vector<int32_t>* shape_vector = const_cast<flatbuffers::Vector<int32_t>*>(shape_vector_const);
    shape_vector->Mutate(shape_index, new_shape);
    if(new_shape != (*shape_vector)[shape_index]) {
      TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: Shape change failed.");
      Serial1.println("ERROR: Shape change failed.");
      return 1;
    }
  }
  return 0;
}

void ModelModifier::modifyFullyConnectedShape(const int32_t layer_index, const int32_t new_shape)
{
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  const tflite::Operator* target_op = (*subgraph->operators())[layer_index];
  if(target_op->opcode_index() != op_index_fully_connected_) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: Layer to modify is not Fully-Connected.");
    Serial1.println("ERROR: Layer to modify is not Fully-Connected.");
    return;
  }
  int32_t target_tensor = getWeightTensorIndex(layer_index);
  int32_t next_target_tensor = getWeightTensorIndex(layer_index + 1);
  if(target_tensor == -1 || next_target_tensor == -1) {
    return;
  }
  int8_t ret = setTensorShape(target_tensor, new_shape);
  ret = setTensorShape(next_target_tensor, new_shape, 1);
}
