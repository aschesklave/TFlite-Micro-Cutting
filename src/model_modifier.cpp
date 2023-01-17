#include <Arduino.h>
#include "model_modifier.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/modifier_params.h"

uint32_t* weight_offset = nullptr;
uint32_t current_layer_index = 0;

ModelModifier::ModelModifier(tflite::Model* model)
  : model_(model)
{
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  weight_offset = (uint32_t*)calloc(subgraph->operators()->size(), sizeof(uint32_t));
}

ModelModifier::~ModelModifier() {
  free(weight_offset);
  weight_offset = nullptr;
}

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

int32_t ModelModifier::getWeightTensorIndex(const uint32_t& target_op_index) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if (target_op_index >= subgraph->operators()->size()) {
    MicroPrintf("ERROR: layer index out of bounds.");
    Serial1.println("ERROR: layer index out of bounds.");
    return -1;
  }
  const tflite::Operator* target_op = (*subgraph->operators())[target_op_index];
  if (subgraph->operators()->size() < 3) {
    MicroPrintf("ERROR: selected layer needs 3 inputs but has %d", subgraph->operators()->size());
    Serial1.println("ERROR: selected layer needs 3 inputs but has "); Serial1.println(subgraph->operators()->size());
    return -1;
  }
  return (*target_op->inputs())[1];
}

int32_t ModelModifier::setTensorShape(const uint32_t tensor_index, const int32_t new_shape, const uint32_t shape_index) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if (tensor_index >= subgraph->tensors()->size()) {
    MicroPrintf("ERROR: tensor index out of bounds.");
    Serial1.println("ERROR: tensor index out of bounds.");
    return -1;
  }
  const tflite::Tensor* tensor = (*subgraph->tensors())[tensor_index];
  const flatbuffers::Vector<int32_t>* shape_vector_const = tensor->shape();
  if (shape_index >= shape_vector_const->size()) {
    MicroPrintf("ERROR: shape index out of bounds.");
    Serial1.println("ERROR: shape index out of bounds.");
    return -1;
  }
  const int32_t old_shape = (*shape_vector_const)[shape_index];
  if (new_shape != old_shape) {
    if(new_shape > old_shape) {
      MicroPrintf("WARNING: Increased shape.");
      Serial1.println("WARNING: Increased shape.");
    }
    flatbuffers::Vector<int32_t>* shape_vector = const_cast<flatbuffers::Vector<int32_t>*>(shape_vector_const);
    shape_vector->Mutate(shape_index, new_shape);
    if(new_shape != (*shape_vector)[shape_index]) {
      MicroPrintf("ERROR: Shape change failed.");
      Serial1.println("ERROR: Shape change failed.");
      return -1;
    }
  }

  if (shape_index == 1) {
    return old_shape - new_shape;
  }
  return 0;
}

void ModelModifier::modifyFullyConnectedShape(const int32_t layer_index, const int32_t new_shape) {
  if(findOpCodeIndex(tflite::BuiltinOperator_FULLY_CONNECTED, op_index_fully_connected_)) {
    MicroPrintf("ERROR: No FULLY_CONNECTED layer found.");
    Serial1.println("ERROR: No FULLY_CONNECTED layer found.");
  }
  //TODO: Check if layer is out of bounds!
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  const tflite::Operator* target_op = (*subgraph->operators())[layer_index];
  if(target_op->opcode_index() != op_index_fully_connected_) {
    MicroPrintf("ERROR: Layer to modify is not Fully-Connected.");
    Serial1.println("ERROR: Layer to modify is not Fully-Connected.");
    return;
  }
  int32_t target_tensor = getWeightTensorIndex(layer_index);
  int32_t next_target_tensor = getWeightTensorIndex(layer_index + 1);
  if(target_tensor < 0 || next_target_tensor < 0) {
    return;
  }
  int32_t res = setTensorShape(target_tensor, new_shape);
  if(res < 0) return;
  res = setTensorShape(next_target_tensor, new_shape, 1);
  if(res < 0) return;
  weight_offset[layer_index + 1] = res;
}

void ModelModifier::modify2DConvolutionalShape(const int32_t layer_index, const int32_t new_shape) {
  if(findOpCodeIndex(tflite::BuiltinOperator_CONV_2D, op_index_2d_convolutional_)) {
    MicroPrintf("ERROR: No CONV_2D layer found.");
    Serial1.println("ERROR: No CONV_2D layer found.");
    const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
    const tflite::Operator* target_op = (*subgraph->operators())[layer_index];
    if(target_op->opcode_index() != op_index_fully_connected_) {
      MicroPrintf("ERROR: Layer to modify is not Conv2D.");
      Serial1.println("ERROR: Layer to modify is not Conv2D.");
      return;
  }
  }
}
