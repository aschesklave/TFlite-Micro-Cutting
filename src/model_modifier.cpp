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
  op_index_fully_connected_ = -1;
  op_index_2d_convolutional_ = -1;
  op_index_reshape_ = -1;

  (void)findOpCodeIndex(tflite::BuiltinOperator_CONV_2D, op_index_2d_convolutional_);
  (void)findOpCodeIndex(tflite::BuiltinOperator_FULLY_CONNECTED, op_index_fully_connected_);
  (void)findOpCodeIndex(tflite::BuiltinOperator_RESHAPE, op_index_reshape_);

  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  weight_offset = (uint32_t*)calloc(subgraph->operators()->size(), sizeof(uint32_t));
}

ModelModifier::~ModelModifier() {
  free(weight_offset);
  weight_offset = nullptr;
}

uint8_t ModelModifier::findOpCodeIndex(const tflite::BuiltinOperator op, int32_t& index) {
  const auto& opcode_vector = model_->operator_codes();
  for(auto it = opcode_vector->begin(); it != opcode_vector->end(); ++it) {
    if(it->builtin_code() == op) {
      index = std::distance(opcode_vector->begin(), it);
      return 0;
    }
  }
  return 1;
}

int32_t ModelModifier::getInputTensorIndex(const uint32_t& target_op_index) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if (target_op_index >= subgraph->operators()->size()) {
    MicroPrintf("ERROR: layer index out of bounds.");
    Serial1.println("ERROR: layer index out of bounds.");
    return -1;
  }
  const tflite::Operator* target_op = (*subgraph->operators())[target_op_index];
  return (*target_op->inputs())[0];
}

int32_t ModelModifier::getOutputTensorIndex(const uint32_t& target_op_index) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if (target_op_index >= subgraph->operators()->size()) {
    MicroPrintf("ERROR: layer index out of bounds.");
    Serial1.println("ERROR: layer index out of bounds.");
    return -1;
  }
  const tflite::Operator* target_op = (*subgraph->operators())[target_op_index];
  return (*target_op->outputs())[0];
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

int8_t ModelModifier::setTensorShape(const int32_t tensor_index, const int32_t new_shape, const uint32_t shape_index, int32_t &shape_diff) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if (tensor_index < 0 || (uint32_t)tensor_index >= subgraph->tensors()->size()) {
    MicroPrintf("ERROR: tensor index out of bounds.");
    Serial1.println("ERROR: tensor index out of bounds.");
    return 1;
  }
  const tflite::Tensor* tensor = (*subgraph->tensors())[tensor_index];
  const flatbuffers::Vector<int32_t>* shape_vector_const = tensor->shape();
  if (shape_index >= shape_vector_const->size()) {
    MicroPrintf("ERROR: shape index out of bounds.");
    Serial1.println("ERROR: shape index out of bounds.");
    return 1;
  }
  const int32_t old_shape = (*shape_vector_const)[shape_index];
  if (new_shape != old_shape) {
    if(new_shape > old_shape) {
      //MicroPrintf("WARNING: Increased shape.");
      //Serial1.print("WARNING: Increased shape.");
    }
    flatbuffers::Vector<int32_t>* shape_vector = const_cast<flatbuffers::Vector<int32_t>*>(shape_vector_const);
    shape_vector->Mutate(shape_index, new_shape);
    if(new_shape != (*shape_vector)[shape_index]) {
      MicroPrintf("ERROR: Shape change failed.");
      Serial1.println("ERROR: Shape change failed.");
      return 1;
    }
  }
  shape_diff = old_shape - new_shape;
  return 0;
}

int32_t ModelModifier::getMultipliedTensorShape(const int32_t tensor_index) {
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if((uint32_t)tensor_index >= subgraph->tensors()->size()) {
    MicroPrintf("ERROR: tensor index out of bounds.");
    Serial1.println("ERROR: tensor index out of bounds.");
    return -1;
  }
  const tflite::Tensor* tensor = (*subgraph->tensors())[tensor_index];
  const flatbuffers::Vector<int32_t>* shape_vector = tensor->shape();
  int32_t result = 1;
  for(uint8_t i = 0; i < shape_vector->size(); ++i) {
    result *= (*shape_vector)[i];
  }
  return result;
}

void ModelModifier::modifyFullyConnectedShape(const int32_t layer_index, const int32_t new_shape) {
  if(op_index_fully_connected_ < 0) {
    MicroPrintf("ERROR: No FULLY_CONNECTED layer found.");
    Serial1.println("ERROR: No FULLY_CONNECTED layer found.");
  }
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if(layer_index < 0 || (uint32_t)layer_index >= subgraph->operators()->size()) {
    MicroPrintf("ERROR: layer index out of bounds.");
    Serial1.println("ERROR: layer index out of bounds.");
    return;
  }
  const tflite::Operator* target_op = (*subgraph->operators())[layer_index];
  if(target_op->opcode_index() != (uint32_t)op_index_fully_connected_) {
    MicroPrintf("ERROR: Layer to modify is not Fully-Connected.");
    Serial1.println("ERROR: Layer to modify is not Fully-Connected.");
    return;
  }
  int32_t target_tensor = getWeightTensorIndex(layer_index);
  int32_t next_target_tensor = getWeightTensorIndex(layer_index + 1);
  if(target_tensor < 0 || next_target_tensor < 0) {
    return;
  }
  int32_t shape_diff;
  int8_t status = setTensorShape(target_tensor, new_shape, 0, shape_diff);
  if(0 == status) {
    status = setTensorShape(next_target_tensor, new_shape, 1, shape_diff);
  }
  if(0 == status) {
    weight_offset[layer_index + 1] += shape_diff;
  }
}

void ModelModifier::modify2DConvolutionalShape(const int32_t layer_index, const int32_t new_shape) {
  if(op_index_2d_convolutional_ < 0) {
    MicroPrintf("ERROR: No CONV_2D layer found.");
    Serial1.println("ERROR: No CONV_2D layer found.");
  }
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];
  if(layer_index < 0 || (uint32_t)layer_index >= subgraph->operators()->size()) {
    MicroPrintf("ERROR: layer index out of bounds.");
    Serial1.println("ERROR: layer index out of bounds.");
    return;
  }
  const tflite::Operator* target_op = (*subgraph->operators())[layer_index];
  if(target_op->opcode_index() != (uint32_t)op_index_2d_convolutional_) {
    MicroPrintf("ERROR: Layer to modify is not Conv2D.");
    Serial1.println("ERROR: Layer to modify is not Conv2D.");
    return;
  }

  int32_t first_conv_weight_tensor = getWeightTensorIndex(layer_index);
  int32_t conv_output_tensor = getOutputTensorIndex(layer_index);
  int32_t pool_output_tensor = getOutputTensorIndex(layer_index + 1);

  int32_t shape_diff;
  int8_t status = setTensorShape(first_conv_weight_tensor, new_shape, 0, shape_diff);

  if (0 == status) {
    status = setTensorShape(conv_output_tensor, new_shape, 3, shape_diff);
  }
  if (0 == status) {
    status = setTensorShape(pool_output_tensor, new_shape, 3, shape_diff);
  }
  int32_t second_layer_index = layer_index + 2;
  const uint32_t second_layer_opcode = (*subgraph->operators())[second_layer_index]->opcode_index();

  if (second_layer_opcode == (uint32_t)op_index_2d_convolutional_) {
    int32_t second_conv_weight_tensor = getWeightTensorIndex(second_layer_index);
    if (0 == status) {
      status = setTensorShape(second_conv_weight_tensor, new_shape, 3, shape_diff);
    }
    if (0 == status) {
      weight_offset[second_layer_index] += shape_diff;
    }
  }
  else if (second_layer_opcode == (uint32_t)op_index_reshape_) {
    int32_t fc_shape = getMultipliedTensorShape(pool_output_tensor);
    int32_t fc_layer_idx = second_layer_index + 1;
    int32_t fc_weight_tensor = getWeightTensorIndex(fc_layer_idx);
    if (0 == status) {
      status = setTensorShape(fc_weight_tensor, fc_shape, 1, shape_diff);
    }
    if (0 == status) {
      weight_offset[fc_layer_idx] += shape_diff;
    }
  }
  else {
    MicroPrintf("ERROR: Unsupported model architecture.");
    Serial1.println("ERROR: Unsupported model architecture.");
  }
}

uint32_t ModelModifier::calcModelParams(bool is_cnn) {
  uint32_t param_count = 0;
  const tflite::SubGraph* subgraph = (*model_->subgraphs())[0];

  if(is_cnn) {

  }
  else {
    for(uint8_t i = 0; i < subgraph->operators()->size() - 1; ++i) {
      const flatbuffers::Vector<int32_t>* weight_shape_vector = (*subgraph->tensors())[getWeightTensorIndex(i)]->shape();
      uint32_t layer_param_count = 1;
      for(auto it = weight_shape_vector->begin(); it != weight_shape_vector->end(); ++it) {
        layer_param_count *= *it;
      }
      param_count += layer_param_count;
    }
  }

  return param_count;
}