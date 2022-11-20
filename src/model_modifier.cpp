#include "model_modifier.h"
#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"

class bypass_private;
template <>
bypass_private* tflite::MicroInterpreter::typed_input_tensor<bypass_private>(int tensor_index) {
  model_ = reinterpret_cast<Model*>(tensor_index);
  return nullptr;
}

ModelModifier::ModelModifier(tflite::MicroInterpreter* interpreter, tflite::Model* unmodified_model, tflite::ErrorReporter* error_reporter)
{
  interpreter_ = interpreter;
  error_reporter_ = error_reporter;
  unpacked_model_ = unmodified_model->UnPack();
  opcodes_ = &unpacked_model_->operator_codes;
  subgraph_ = (unpacked_model_->subgraphs[0]).get();

  if(findOpCodeIndex(tflite::BuiltinOperator_FULLY_CONNECTED, op_index_fully_connected_)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: No FULLY_CONNECTED layer found.");
  }
}

ModelModifier::~ModelModifier()
{
}

uint8_t ModelModifier::findOpCodeIndex(const tflite::BuiltinOperator op, uint32_t& index) {
  for(auto it = opcodes_->begin(); it != opcodes_->end(); ++it) {
    if((*it)->builtin_code == op) {
      index = std::distance(opcodes_->begin(), it);
      return 0;
    }
  }
  return 1;
}

int32_t ModelModifier::getWeightTensorIndex(const int32_t& target_op_index) {
  return subgraph_->operators[target_op_index]->inputs[1];
}

uint8_t ModelModifier::setTensorShape(const int32_t tensor_index, const int32_t new_shape) {
  auto& shape = subgraph_->tensors[tensor_index]->shape;
  if(new_shape >= shape[0]) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: Attempted to increase shape.");
    return 1;
  }
  shape[0] = new_shape;
  return 0;
}

void ModelModifier::modifyShape(const int32_t layer_index, const int32_t new_shape)
{
  const auto& operators = subgraph_->operators;
  tflite::OperatorT* target_op = operators[layer_index].get();
  if(target_op->opcode_index != op_index_fully_connected_) {
    TF_LITE_REPORT_ERROR(error_reporter_, "ERROR: Layer to modify is not Fully-Connected.");
    return;
  }

  int32_t target_tensor = getWeightTensorIndex(layer_index);
  int8_t ret = setTensorShape(target_tensor, new_shape);

  static char inst_memory[sizeof(flatbuffers::FlatBufferBuilder)];
  // Make member
  flatbuffers::FlatBufferBuilder* fbb =
      new (inst_memory) flatbuffers::FlatBufferBuilder(
          8192,
          &CustomStackAllocator::instance(16));

  auto model_offset = tflite::Model::Pack(*fbb, unpacked_model_);

  tflite::FinishModelBuffer(*fbb, model_offset);
  void* model_pointer = fbb->GetBufferPointer();
  const tflite::Model* tmp_model = flatbuffers::GetRoot<tflite::Model>(model_pointer);
  tflite::Model* new_model = const_cast<tflite::Model*>(tmp_model);

  interpreter_->typed_input_tensor<bypass_private>((int)new_model);
  delete fbb;
}
