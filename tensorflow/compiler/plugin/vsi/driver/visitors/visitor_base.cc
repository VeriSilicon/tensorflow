/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/vsi/driver/visitors/visitor_base.h"

#include <stddef.h>

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/compiler/plugin/vsi/driver/custom_kernels_util.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops.h"

namespace xla {
namespace vsiplugin {

bool IsRootHlo(const HloInstruction* hlo) {
  bool is_root = false;
#if 1
  if (hlo->users().size() > 0) {
    std::vector<HloOpcode> mpi_list = {HloOpcode::kAllReduce};
    if (std::any_of(mpi_list.begin(), mpi_list.end(), [hlo](HloOpcode op) {
          return op == hlo->users()[0]->opcode();
        })) {
      // LOG(INFO) << "FFFTT IsRootHlo: mpi_list";
      return true;
    }
  }

  const HloInstruction* root = hlo->parent()->root_instruction();
  std::vector<HloOpcode> except_list = {HloOpcode::kGetTupleElement,
                                        HloOpcode::kTuple, HloOpcode::kCopy};
  if (std::none_of(except_list.begin(), except_list.end(),
                   [root](HloOpcode op) { return op == root->opcode(); })) {
    return (root == hlo);
  }

  auto root1 = root;
  if (root->opcode() == HloOpcode::kGetTupleElement) {
    root1 = root->operand(0);
  }
  for (auto operand : root1->operands()) {
    const HloInstruction* operand1 = operand;
    while (operand1->opcode() == HloOpcode::kCopy) {
      operand1 = operand1->operand(0);
    }
    if (operand1 == hlo) {
      return true;
    }
  }
#endif
  return is_root;
}

Literal BaseVisitor::evaluate(
    const HloComputation& computation
    /*absl::Span<const Literal* const> arg_literals*/) {
  computation.Accept(this);
  return GetEvaluatedLiteralFor(computation.root_instruction()).Clone();
}

std::vector<std::shared_ptr<tim::vx::Tensor>> BaseVisitor::evaluate(
    const HloComputation& computation,
    std::vector<Literal>& argument_literals) {
  arg_literals_ = std::move(argument_literals);
  auto input_tensors = graph_->InputsTensor();

  if (is_build_ && !arg_literals_.empty()) {
    // input_tensors include not only mutable parameters(arg_literals_) but also
    // const parameters.
    LOG(INFO) << "BaseVisitor::evaluate 0";
    CHECK_LE(arg_literals_.size(), input_tensors.size());

    for (uint32_t i = 0; i < arg_literals_.size(); i++) {
      auto& input_literal = arg_literals_[i];
      uint32_t input_id = kVsiInputId_[static_cast<int64_t>(i)];

      for (auto input_tensor : input_tensors) {
        if (input_id == input_tensor->GetId()) {
          ShapeIndex shapeIndex({});
          void* buffer = input_literal.untyped_data(shapeIndex);

          input_tensor->CopyDataToTensor(buffer);
          break;
        }
      }
    }
  } else {
    LOG(INFO) << "BaseVisitor::evaluate 1";
    computation.Accept(this);
  }
  graph_->PrintGraph();
  std::vector<std::shared_ptr<tim::vx::Tensor>> fault_result;
  if (!graph_->Compile()) {
    LOG(FATAL) << "Compile graph fail.";
    return fault_result;
  }
  if (!graph_->Run()) {
    LOG(FATAL) << "Run graph fail";
    return fault_result;
  }
  is_build_ = true;

  return GetEvaluatedTensorFor(computation.root_instruction());
}

const Shape& BaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

/*the function should only be used by Handlexxxxx function so that the tensor
  maped to {$/hlo/$} has been created.
  the order of tensor's layout is the same as its shape: minjor to major   */
std::shared_ptr<tim::vx::Tensor> BaseVisitor::InsertTranspose(
    const HloInstruction* hlo, std::vector<uint32_t>& dim_index) {
  auto shape = hlo->shape();
  size_t dim_size = dim_index.size();
  std::vector<int64_t> output_dims(dim_size, 1);
  std::vector<uint32_t> perm(dim_size, 1);

  auto input_tensor = GetEvaluatedTensorFor(hlo)[0];

  /*check if the layout is {WHCN} , if not, a transpose would be inserted to
   * covert the layout. */
  bool is_need_insert_transpose = false;
  for (int i = 0; i < dim_size; i++) {
    if (dim_index[i] == shape.layout().minor_to_major()[dim_size - i - 1]) {
      perm[dim_size - 1 - i] = dim_size - i - 1;
    } else {
      is_need_insert_transpose = true;
      for (int j = 0; j < dim_size; j++) {
        if (dim_index[i] != shape.layout().minor_to_major()[j]) continue;
        perm[dim_size - 1 - i] = j;
        break;
      }
    }
  }
  std::ostringstream ss, ss1, ss2, ss3;
  for (int i = 0; i < dim_size; i++) {
    ss << dim_index[i] << " ";
    ss1 << shape.layout().minor_to_major()[i] << " ";
    ss2 << perm[i] << " ";
    ss3 << shape.dimensions(i) << " ";
  }
  LOG(INFO) << "InsertTranspose 0: " << is_need_insert_transpose << " : "
            << dim_size;
  LOG(INFO) << "InsertTranspose 1: dim_index: " << ss.str();
  LOG(INFO) << "InsertTranspose 2: minor_to_major: " << ss1.str();
  LOG(INFO) << "InsertTranspose 3: perm: " << ss2.str();
  LOG(INFO) << "InsertTranspose 4: hlo->shape: " << ss3.str();

  if (is_need_insert_transpose) {
    LOG(INFO) << "InsertTranspose 5: ";
    auto input_shape = input_tensor->GetShape();
    std::vector<uint32_t> output_shape;
    for (auto d : perm) {
      output_shape.push_back(input_shape[d]);
    }
    auto output_tensor = CreateTensorFromShape(
        convertTfPrimitiveTypeToTim(hlo->shape().element_type()), output_shape,
        tim::vx::TensorAttribute::OUTPUT);
    auto transposeOp = graph_->CreateOperation<tim::vx::ops::Transpose>(perm);
    transposeOp->BindInput(input_tensor).BindOutput(output_tensor);

    return output_tensor;
  }
  return input_tensor;
}

template <typename T>
Status BaseVisitor::HandleSimpleElementwiseUnary(HloInstruction* hlo) {
  auto shape = hlo->shape();
  const HloInstruction* input = hlo->operand(0);
  TF_RET_CHECK(ShapeUtil::SameDimensions(shape, input->shape()));
  auto input_tensor = GetEvaluatedTensorFor(input)[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  auto op = graph_->CreateOperation<T>();
  op->BindInput(input_tensor).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleElementwiseUnary(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__ << " : "
            << HloOpcodeString(hlo->opcode());
  switch (hlo->opcode()) {
    case HloOpcode::kNegate:
      HandleSimpleElementwiseUnary<tim::vx::ops::Neg>(hlo);
      break;
    case HloOpcode::kExp:
      HandleSimpleElementwiseUnary<tim::vx::ops::Exp>(hlo);
      break;
    case HloOpcode::kLog:
      HandleSimpleElementwiseUnary<tim::vx::ops::Log>(hlo);
      break;
    case HloOpcode::kNot:
      HandleSimpleElementwiseUnary<tim::vx::ops::LogicalNot>(hlo);
      break;
    case HloOpcode::kSqrt:
      HandleSimpleElementwiseUnary<tim::vx::ops::Sqrt>(hlo);
      break;
case HloOpcode::kRsqrt:
      HandleSimpleElementwiseUnary<tim::vx::ops::Rsqrt>(hlo);
      break;
    case HloOpcode::kCeil:
      HandleSimpleElementwiseUnary<tim::vx::ops::Ceil>(hlo);
      break;
    case HloOpcode::kAbs:
      HandleSimpleElementwiseUnary<tim::vx::ops::Abs>(hlo);
      break;
    case HloOpcode::kTanh:
      HandleSimpleElementwiseUnary<tim::vx::ops::Tanh>(hlo);
      break;
    case HloOpcode::kRoundNearestAfz:
      HandleSimpleElementwiseUnary<tim::vx::ops::Round>(hlo);
      break;
    case HloOpcode::kFloor:
      HandleSimpleElementwiseUnary<tim::vx::ops::Floor>(hlo);
      break;
    case HloOpcode::kSin:
      HandleSimpleElementwiseUnary<tim::vx::ops::Sin>(hlo);
      break;

    default:
      LOG(INFO) << "has not been implement; opcode:"
                << HloOpcodeString(hlo->opcode());
      return tensorflow::errors::Unimplemented(
          "some HandleElementwiseUnary op has not been implement");
  }
  return Status::OK();
}

template <typename T>
Status BaseVisitor::HandleSimpleElementwiseBinary(HloInstruction* hlo) {
  auto shape = hlo->shape();
  const HloInstruction* lhs = hlo->operand(0);
  const HloInstruction* rhs = hlo->operand(1);
  TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
  TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  auto lhs_tensor = GetEvaluatedTensorFor(lhs)[0];
  auto rhs_tensor = GetEvaluatedTensorFor(rhs)[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);
  auto op = graph_->CreateOperation<T>();
  op->BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleElementwiseBinary(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__ << " : "
            << HloOpcodeString(hlo->opcode());
  switch (hlo->opcode()) {
    case HloOpcode::kAdd: {
      HandleSimpleElementwiseBinary<tim::vx::ops::Add>(hlo);
      break;
    }
    case HloOpcode::kSubtract: {
      HandleSimpleElementwiseBinary<tim::vx::ops::Sub>(hlo);
      break;
    }
    case HloOpcode::kMultiply: {
      auto shape = hlo->shape();
      const HloInstruction* lhs = hlo->operand(0);
      const HloInstruction* rhs = hlo->operand(1);
      auto left_shape = lhs->shape();
      auto right_shape = rhs->shape();
      TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
      // TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

      tim::vx::ShapeType left_timShape;
      if (left_shape.is_static() && left_shape.has_layout()) {
        for (auto d : left_shape.layout().minor_to_major())
          left_timShape.push_back(left_shape.dimensions(d));
      }

      tim::vx::ShapeType right_timShape;
      if (right_shape.is_static() && right_shape.has_layout()) {
        for (auto d : right_shape.layout().minor_to_major())
          right_timShape.push_back(right_shape.dimensions(d));
      }

      if (left_timShape.size() == 2 && right_timShape.size() == 2 &&
          left_timShape[1] != right_timShape[1]) {
        auto lhs_tensor = GetEvaluatedTensorFor(rhs)[0];
        auto rhs_tensor = GetEvaluatedTensorFor(lhs)[0];
        auto out_tensor = CreateTensorFromShape(
            shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                    : tim::vx::TensorAttribute::TRANSIENT);
        auto mul = graph_->CreateOperation<tim::vx::ops::Multiply>();
        mul->BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

        vsi_run_tensor_container_[hlo].push_back(out_tensor);
        break;
      }

      auto lhs_tensor = GetEvaluatedTensorFor(lhs)[0];
      auto rhs_tensor = GetEvaluatedTensorFor(rhs)[0];
      auto out_tensor = CreateTensorFromShape(
          shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                  : tim::vx::TensorAttribute::TRANSIENT);
      auto mul = graph_->CreateOperation<tim::vx::ops::Multiply>();
      mul->BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

      vsi_run_tensor_container_[hlo].push_back(out_tensor);
      break;
    }
    case HloOpcode::kDivide: {
      auto shape = hlo->shape();
      const HloInstruction* lhs = hlo->operand(0);
      const HloInstruction* rhs = hlo->operand(1);
      TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));

      auto lhs_tensor = GetEvaluatedTensorFor(lhs)[0];
      auto rhs_tensor = GetEvaluatedTensorFor(rhs)[0];
      auto out_tensor = CreateTensorFromShape(
          shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                  : tim::vx::TensorAttribute::TRANSIENT);
      auto div = graph_->CreateOperation<tim::vx::ops::Div>();
      if (rhs_tensor->IsConstTensor() ||
          rhs_tensor->GetDataType() == tim::vx::DataType::INT32) {
        auto timShape = rhs_tensor->GetShape();
        auto dataType = rhs_tensor->GetDataType();
        auto spec = rhs_tensor->GetSpec();

        tim::vx::TensorSpec timSpec(tim::vx::DataType::FLOAT32, timShape,
                                    tim::vx::TensorAttribute::INPUT);

        uint32_t size = 1;
        for (uint32_t i = 0; i < timShape.size(); i++) {
          size = size * timShape[i];
        }
        size = size * 4;

        int32_t buffer_data;
        rhs_tensor->CopyDataFromTensor(static_cast<void*>(&buffer_data));
        float buffer_data_transform = (float)buffer_data;

        auto rhs_tensor_transform = graph_->CreateTensor(
            timSpec, static_cast<void*>(&buffer_data_transform));
        div->BindInput(lhs_tensor)
            .BindInput(rhs_tensor_transform)
            .BindOutput(out_tensor);

      } else {
        div->BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);
      }
      vsi_run_tensor_container_[hlo].push_back(out_tensor);
      break;
    }
    case HloOpcode::kMaximum: {
      HandleSimpleElementwiseBinary<tim::vx::ops::Maximum>(hlo);
      break;
    }
    case HloOpcode::kMinimum: {
      HandleSimpleElementwiseBinary<tim::vx::ops::Minimum>(hlo);
      break;
    }
    case HloOpcode::kPower: {
      HandleSimpleElementwiseBinary<tim::vx::ops::Pow>(hlo);
      break;
    }
    // case HloOpcode::kRemainder: {
    //   return dovisitor->HandleElementwiseBinaryRemainder(
    //       hlo, mutex_, vsi_run_tensor_container_, graph_);
    //   break;
    // }
    case HloOpcode::kAnd: {
      HandleSimpleElementwiseBinary<tim::vx::ops::LogicalAnd>(hlo);
      break;
    }
    case HloOpcode::kOr: {
      HandleSimpleElementwiseBinary<tim::vx::ops::LogicalOr>(hlo);
      break;
    }
    default:
      LOG(INFO) << "has not been implement; opcode:"
                << HloOpcodeString(hlo->opcode());
      return tensorflow::errors::Unimplemented(
          "some HandleElementwiseBinary op has not been implement");
  }
  return Status::OK();
}

Status BaseVisitor::FinishVisit(HloInstruction* root) { return Status::OK(); }

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return xla::Unimplemented("%s (%s) not implemented", inst->name().c_str(),
                            HloOpcodeString(inst->opcode()).c_str());
}

Status BaseVisitor::HandleConvert(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  const HloInstruction* input = hlo->operand(0);
  auto shape = hlo->shape();
  auto in_tensor = GetEvaluatedTensorFor(input)[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  auto op = graph_->CreateOperation<tim::vx::ops::Cast>();
  (*op).BindInput(in_tensor).BindOutput(out_tensor);
  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandlePad(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;

  auto shape = hlo->shape();
  auto* pad_hlo = Cast<HloPadInstruction>(hlo);

  const HloInstruction* t = hlo->operand(0);  // t
  const HloInstruction* padding_value =
      pad_hlo->padding_value();  // padding_value
  const PaddingConfig& padding_config = pad_hlo->padding_config();

  auto t_tensor = GetEvaluatedTensorFor(t)[0];
  auto padding_value_tensor = GetEvaluatedTensorFor(padding_value)[0];

  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  {
    std::ostringstream ss;
    auto dims = padding_config.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].edge_padding_low() << " " << dims[i].edge_padding_high() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " pad: " << ss.str();
  }

  std::vector<uint32_t> front;
  std::vector<uint32_t> back;

  auto pad_dims = padding_config.dimensions();
  for (int i = 0; i < pad_dims.size(); i++) {
    front.push_back(pad_dims[i].edge_padding_low());
    back.push_back(pad_dims[i].edge_padding_high());
  }

  auto op = graph_->CreateOperation<tim::vx::ops::Pad>(
      front, back, 0, tim::vx::ops::Pad::pad_mode_type::PAD_MODE_CONSTANT);

  (*op).BindInput(t_tensor).BindOutput(out_tensor);
  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleTuple(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__
            << " : operand count : " << hlo->operand_count();

  auto shape = hlo->shape();
  int64_t input_num = hlo->operand_count();
  for (int64_t i = 0; i < input_num; i++) {
    const HloInstruction* input = hlo->operand(i);
    LOG(INFO) << "opcode : " << HloOpcodeString(input->opcode());
    auto it = vsi_run_tensor_container_.find(input);
    {
      std::ostringstream ss;
      auto shape = it->second[0]->GetSpec().shape_;
      for (auto size : shape) {
        ss << size << " ";
      }
      LOG(INFO) << __FUNCTION__ << " shape : " << ss.str();
    }
    vsi_run_tensor_container_[hlo].push_back(it->second[0]);
  }

  // auto it = vsi_run_tensor_container_.find(input);
  // vsi_run_tensor_container_[hlo] = it->second;
  return Status::OK();
}

Status BaseVisitor::HandleGetTupleElement(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;

  auto* tuple_hlo = Cast<HloGetTupleElementInstruction>(hlo);
  int64_t index = tuple_hlo->tuple_index();

  LOG(INFO) << "tuple_index : " << index << " :: " << hlo->operand_count();
  for (int64_t i = 0; i < hlo->operand_count(); i++) {
    const HloInstruction* input = hlo->operand(i);
    LOG(INFO) << "opcode : " << HloOpcodeString(input->opcode());
  }

  LOG(INFO) << "PROCESS 1 " << __FUNCTION__;
  const HloInstruction* input = hlo->operand(0);
  LOG(INFO) << "PROCESS 2 " << __FUNCTION__;
  auto it = vsi_run_tensor_container_.find(input);
  if (it == vsi_run_tensor_container_.end()) {
    LOG(INFO) << "PROCESS FUCK ,can not find " << __FUNCTION__;
    return Status::OK();
  }
  LOG(INFO) << "PROCESS 3 " << __FUNCTION__;

  vsi_run_tensor_container_[hlo].push_back(it->second[index]);
  LOG(INFO) << "PROCESS 4 " << __FUNCTION__;

  // auto shape = hlo->shape();

  // const HloInstruction* input = hlo->operand(tuple_hlo);
  // auto it = vsi_run_tensor_container_.find(input);
  // vsi_run_tensor_container_[hlo] = it->second;
  return Status::OK();
}

Status BaseVisitor::HandleCopy(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  const HloInstruction* input = hlo->operand(0);

  auto it = vsi_run_tensor_container_.find(input);
  if (it == vsi_run_tensor_container_.end()) {
    LOG(INFO) << "PROCESS FUCK ,can not find " << __FUNCTION__;
    return Status::OK();
  }
  // HandleCopy is cooperate with output tuple.
  // In VSI backend, we always create new tensor for output tensor,
  // so data copy is not necessary on our backend.
  // Just reserve some codes for debug usage.
#if 0
    auto in_tensor = GetEvaluatedTensorFor(input);
    auto out_tensor = CreateTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);

    auto convert = graph_->CreateOperation<tim::vx::ops::DataConvert>();
    convert->BindInput(in_tensor[0]).BindOutput(out_tensor);

    vsi_run_tensor_container_[hlo].push_back(out_tensor);
#else
  vsi_run_tensor_container_[hlo].push_back(it->second[0]);
#endif
  // LOG(INFO) << "PROCESS 4 " << __FUNCTION__;

  return Status::OK();
}

template <typename T>
Status BaseVisitor::CreateReduceOp(std::shared_ptr<tim::vx::Tensor>& input,
                                   std::shared_ptr<tim::vx::Tensor>& output,
                                   std::vector<int32_t>& axis) {
  auto reduce = graph_->CreateOperation<T>(axis, false);
  reduce->BindInput(input).BindOutput(output);
  return Status::OK();
}

Status BaseVisitor::HandleReduceOpMap(HloOpcode opcode,
                                      std::shared_ptr<tim::vx::Tensor>& input,
                                      std::shared_ptr<tim::vx::Tensor>& output,
                                      std::vector<int32_t>& axis) {
  switch (opcode) {
    case HloOpcode::kAdd:
      CreateReduceOp<tim::vx::ops::ReduceSum>(input, output, axis);
      break;
    case HloOpcode::kMultiply:
      CreateReduceOp<tim::vx::ops::ReduceProd>(input, output, axis);
      break;
    case HloOpcode::kMaximum:
      CreateReduceOp<tim::vx::ops::ReduceMax>(input, output, axis);
      break;
    case HloOpcode::kMinimum:
      CreateReduceOp<tim::vx::ops::ReduceMin>(input, output, axis);
      break;
    case HloOpcode::kAnd:
      CreateReduceOp<tim::vx::ops::ReduceAll>(input, output, axis);
      break;
    case HloOpcode::kOr:
      CreateReduceOp<tim::vx::ops::ReduceAny>(input, output, axis);
      break;
    default:
      return tensorflow::errors::Unimplemented("Unimplemented Compare Op: %s",
                                               HloOpcodeString(opcode).c_str());
  }
  return Status::OK();
}

Status BaseVisitor::HandleReduce(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* reduce_hlo = Cast<HloReduceInstruction>(hlo);

  // CHECK_EQ(reduce_hlo->input_count(), 1);
  int64_t input_num = reduce_hlo->input_count();
  LOG(INFO) << "HandleReduce input_count: " << input_num;
  auto opcode = hlo->to_apply()->root_instruction()->opcode();
  LOG(INFO) << "HandleReduce opcode: " << HloOpcodeString(opcode);

  {
    // Note: init_values is unsupported now.
    std::ostringstream ss;
    auto dims = reduce_hlo->init_values();
    for (int i = 0; i < dims.size(); i++) {
      if (dims[i] != nullptr) {
        ss << dims[i]->ToString() << " : ";
      }
    }
    LOG(INFO) << "HandleReduce init_values: " << ss.str();
  }

  if (input_num == 1) {
    const HloInstruction* input = reduce_hlo->operand(0);
    auto input_tensor = GetEvaluatedTensorFor(input)[0];

    uint32_t input_tensor_dimensions = input_tensor->GetShape().size();

    auto dimensions = hlo->dimensions();
    {
      std::ostringstream ss;
      for (int i = 0; i < dimensions.size(); i++) {
        ss << dimensions[i] << " ";
      }
      LOG(INFO) << "HandleReduce dimension: " << ss.str();
    }

    std::vector<int32_t> axis;
    for (uint32_t i = 0; i < dimensions.size(); i++) {
      axis.push_back(
          static_cast<int32_t>(input_tensor_dimensions - 1 - dimensions[i]));
    }

    auto out_tensor = CreateTensorFromShape(
        shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                : tim::vx::TensorAttribute::TRANSIENT);
    HandleReduceOpMap(opcode, input_tensor, out_tensor, axis);
    vsi_run_tensor_container_[hlo].push_back(out_tensor);
  } else {
    for (int64_t i = 0; i < input_num; i++) {
      const HloInstruction* input = reduce_hlo->operand(i);
      auto input_tensor = GetEvaluatedTensorFor(input)[0];

      uint32_t input_tensor_dimensions = input_tensor->GetShape().size();
      {
        std::ostringstream ss;
        auto dims = input_tensor->GetShape();
        for (int i = 0; i < dims.size(); i++) {
          ss << dims[i] << " ";
        }
        LOG(INFO) << "HandleReduce inputsize: " << ss.str();
      }

      auto dimensions = hlo->dimensions();
      {
        std::ostringstream ss;
        for (int i = 0; i < dimensions.size(); i++) {
          ss << dimensions[i] << " ";
        }
        LOG(INFO) << "HandleReduce dimension: " << ss.str();
      }

      std::vector<int32_t> axis;
      for (uint32_t i = 0; i < dimensions.size(); i++) {
        axis.push_back(
            static_cast<int32_t>(input_tensor_dimensions - 1 - dimensions[i]));

        auto out_tensor = CreateTensorFromTupleShape(
            shape, i, tim::vx::TensorAttribute::OUTPUT);
        HandleReduceOpMap(opcode, input_tensor, out_tensor, axis);
        vsi_run_tensor_container_[hlo].push_back(out_tensor);
      }
    }
  }

  return Status::OK();
}

static tim::vx::PoolType GetPoolType(HloOpcode opcode) {
  tim::vx::PoolType reduction_type = tim::vx::PoolType::MAX;
  switch (opcode) {
    case HloOpcode::kMaximum:
      reduction_type = tim::vx::PoolType::MAX;
      break;
    default: {
      LOG(INFO) << "Unsupported opcode for pool type.";
    }
  }
  return reduction_type;
}

Status BaseVisitor::HandleReduceWindow(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* reduce_window_hlo = Cast<HloReduceWindowInstruction>(hlo);
  const auto& window = hlo->window();
  const HloInstruction* input = hlo->operand(0);

  auto opcode = hlo->to_apply()->root_instruction()->opcode();
  LOG(INFO) << "HandleReduceWindow opcode: " << HloOpcodeString(opcode);

  if (shape.dimensions().size() != 4) {
    return tensorflow::errors::Unimplemented("Only support pool2d.");
  }

  {
    // Note: init_values is unsupported now.
    std::ostringstream ss;
    auto dims = reduce_window_hlo->init_values();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i]->ToString() << " : ";
    }
    LOG(INFO) << __FUNCTION__ << " init_values: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].size() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " size: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].stride() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " stride: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].window_dilation() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " window_dilation: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].base_dilation() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " base_dilation: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].padding_low() << " " << dims[i].padding_high() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " pad: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].size() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " ksize: " << ss.str();
  }

  std::ostringstream ss;
  auto input_minor_to_major = input->shape().layout().minor_to_major();
  for (int i = 0; i < input_minor_to_major.size(); i++) {
    ss << input_minor_to_major[i] << " ";
  }
  LOG(INFO) << __FUNCTION__ << " input_minor_to_major: " << ss.str();

  auto dims = window.dimensions();
  std::array<uint32_t, 2> ksize = {dims[2].size(), dims[1].size()};
  std::array<uint32_t, 2> stride = {dims[2].stride(), dims[1].stride()};
  std::array<uint32_t, 4> pad = {dims[2].padding_low(), dims[2].padding_high(),
                                 dims[1].padding_low(), dims[1].padding_high()};
  auto pool_type = GetPoolType(opcode);

  auto in_tensor = GetEvaluatedTensorFor(hlo->operand(0))[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  std::vector<uint32_t> in_perm;
  std::vector<uint32_t> out_perm;
  if (ss.str() == "3 2 1 0 ") {
    in_perm = {1, 2, 0, 3};   // CWHN -> WHCN
    out_perm = {2, 0, 1, 3};  // WHCN -> CWHN
  } else if (ss.str() == "2 1 3 0 ") {
    in_perm = {0, 1, 2, 3};   // WHCN -> WHCN
    out_perm = {0, 1, 2, 3};  // WHCN -> WHCN
  } else {
    LOG(FATAL) << "PROCESS " << __FUNCTION__
               << " Unsupported Layout:" << ss.str();
  }
  auto in_tensor_spec = in_tensor->GetSpec();
  auto in_tensor_shape = in_tensor_spec.GetShapeType();

  std::vector<uint32_t> in_tensor_tmp_shape = {
      in_tensor_shape[in_perm[0]], in_tensor_shape[in_perm[1]],
      in_tensor_shape[in_perm[2]], in_tensor_shape[in_perm[3]]};
  tim::vx::TensorSpec in_tensor_tmp_sec(in_tensor_spec.GetDataType(),
                                        in_tensor_tmp_shape,
                                        tim::vx::TensorAttribute::TRANSIENT);
  auto in_tensor_tmp = graph_->CreateTensor(in_tensor_tmp_sec);

  auto out_tensor_spec = out_tensor->GetSpec();
  auto out_tensor_shape = out_tensor_spec.GetShapeType();

  std::vector<uint32_t> out_tensor_tmp_shape = {
      out_tensor_shape[in_perm[0]], out_tensor_shape[in_perm[1]],
      out_tensor_shape[in_perm[2]], out_tensor_shape[in_perm[3]]};
  tim::vx::TensorSpec out_tensor_tmp_sec(out_tensor_spec.GetDataType(),
                                         out_tensor_tmp_shape,
                                         tim::vx::TensorAttribute::TRANSIENT);
  auto out_tensor_tmp = graph_->CreateTensor(out_tensor_tmp_sec);

  auto in_transpose = graph_->CreateOperation<tim::vx::ops::Transpose>(in_perm);
  in_transpose->BindInput(in_tensor).BindOutput(in_tensor_tmp);

  auto op = graph_->CreateOperation<tim::vx::ops::Pool2d>(pool_type, pad, ksize,
                                                          stride);
  op->BindInput(in_tensor_tmp).BindOutput(out_tensor_tmp);

  auto out_transpose =
      graph_->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_transpose->BindInput(out_tensor_tmp).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

template <typename T>
Status BaseVisitor::CreateCompareOp(
    std::shared_ptr<tim::vx::Tensor>& lhs_tensor,
    std::shared_ptr<tim::vx::Tensor>& rhs_tensor,
    std::shared_ptr<tim::vx::Tensor>& out_tensor) {
  auto compare = graph_->CreateOperation<T>();
  compare->BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleCompare(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* compare_hlo = Cast<HloCompareInstruction>(hlo);
  const HloInstruction* lhs = hlo->operand(0);
  const HloInstruction* rhs = hlo->operand(1);
  TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
  TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  LOG(INFO) << "HandleCompare direction: " << (int)(compare_hlo->direction());

  auto lhs_tensor = GetEvaluatedTensorFor(lhs)[0];
  auto rhs_tensor = GetEvaluatedTensorFor(rhs)[0];
  auto out_tensor =
      CreateTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
  switch (compare_hlo->direction()) {
    case ComparisonDirection::kEq:
      CreateCompareOp<tim::vx::ops::Equal>(lhs_tensor, rhs_tensor, out_tensor);
      break;
    case ComparisonDirection::kNe:
      CreateCompareOp<tim::vx::ops::NotEqual>(lhs_tensor, rhs_tensor,
                                              out_tensor);
      break;
    case ComparisonDirection::kGe:
      CreateCompareOp<tim::vx::ops::GreaterOrEqual>(lhs_tensor, rhs_tensor,
                                                    out_tensor);
      break;
    case ComparisonDirection::kGt:
      CreateCompareOp<tim::vx::ops::Greater>(lhs_tensor, rhs_tensor,
                                             out_tensor);
      break;
    case ComparisonDirection::kLe:
      CreateCompareOp<tim::vx::ops::LessOrEqual>(lhs_tensor, rhs_tensor,
                                                 out_tensor);
      break;
    case ComparisonDirection::kLt:
      CreateCompareOp<tim::vx::ops::Less>(lhs_tensor, rhs_tensor, out_tensor);
      break;
    default:
      return tensorflow::errors::Unimplemented("Unimplemented Compare Op: %d",
                                               (uint8)compare_hlo->direction());
  }

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleSelect(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  const HloInstruction* condition = hlo->operand(0);
  const HloInstruction* lhs = hlo->operand(1);
  const HloInstruction* rhs = hlo->operand(2);
  TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
  TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  auto condition_tensor = GetEvaluatedTensorFor(condition)[0];
  auto lhs_tensor = GetEvaluatedTensorFor(lhs)[0];
  auto rhs_tensor = GetEvaluatedTensorFor(rhs)[0];
  auto out_tensor =
      CreateTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
  auto select = graph_->CreateOperation<tim::vx::ops::Select>();
  select->BindInput(condition_tensor)
      .BindInput(lhs_tensor)
      .BindInput(rhs_tensor)
      .BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleBroadcast(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto* broadcast_hlo = Cast<HloBroadcastInstruction>(hlo);
  const HloInstruction* input = hlo->operand(0);

  {
    std::ostringstream ss;
    auto dims = input->shape().dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << " HandleBroadcast input shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = hlo->shape().dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << " HandleBroadcast output shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = broadcast_hlo->dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << " HandleBroadcast dimensions 0: " << ss.str();
  }

#if 1
  auto shape = hlo->shape();
  auto in_tensor = GetEvaluatedTensorFor(input)[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  std::vector<uint32_t> out_shape(out_tensor->GetShape().begin(),
                                 out_tensor->GetShape().end());

  std::vector<int32_t> dimensions;
  for (const auto& e : broadcast_hlo->dimensions()) {
    int32_t v = shape.dimensions().size() - 1 - e;
    dimensions.push_back(v);
  }
  {
    std::ostringstream ss;
    for (int i = 0; i < dimensions.size(); i++) {
      ss << dimensions[i] << " ";
    }
    LOG(INFO) << " HandleBroadcast dimensions 1: " << ss.str();
  }
  auto op =
      graph_->CreateOperation<tim::vx::ops::Broadcast>(out_shape, dimensions);
  op->BindInput(in_tensor).BindOutput(out_tensor);
  vsi_run_tensor_container_[hlo].push_back(out_tensor);
#else
  auto it = vsi_run_tensor_container_.find(input);
  vsi_run_tensor_container_[hlo] = it->second;
#endif

  return Status::OK();
}

Status BaseVisitor::HandleScatter(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  // hlo->operand(0) is zeros_like out_tensor
  auto indices_tensor = GetEvaluatedTensorFor(hlo->operand(1))[0];
  auto updates_tensor = GetEvaluatedTensorFor(hlo->operand(2))[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  {
    std::ostringstream ss;
    auto shape = indices_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " indices shape in TIM-VX: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto shape = updates_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " updates shape in TIM-VX: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto shape = out_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " output shape in TIM-VX: " << ss.str();
  }

  auto scatterOp =
      graph_->CreateOperation<tim::vx::ops::ScatterND>(out_tensor->GetShape());

  scatterOp->BindInputs({indices_tensor, updates_tensor})
      .BindOutput(out_tensor);
  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleSelectAndScatter(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  const auto& window = hlo->window();

  if (shape.dimensions().size() != 4) {
    return tensorflow::errors::Unimplemented("Only support pool2d.");
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].size() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " size: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].stride() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " stride: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].padding_low() << " " << dims[i].padding_high() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " pad: " << ss.str();
  }

  std::ostringstream ss;
  auto LAYOUT = hlo->operand(0)->shape().layout().ToString();
  LOG(INFO) << __FUNCTION__ << " LAYOUT: " << LAYOUT; 
  auto input_minor_to_major = 
      hlo->operand(0)->shape().layout().minor_to_major();
  for (int i = 0; i < input_minor_to_major.size(); i++) {
    ss << input_minor_to_major[i] << " ";
  }
  LOG(INFO) << __FUNCTION__ << " input_minor_to_major: " << ss.str(); 
  std::vector<uint32_t> in_perm;
  std::vector<uint32_t> out_perm;
  std::array<uint32_t, 2> ksize;
  std::array<uint32_t, 2> stride;
  auto dims = window.dimensions();
  if (ss.str() == "3 2 1 0 "){
    in_perm = {1, 2, 0, 3};   // CWHN -> WHCN
    out_perm = {2, 0, 1, 3};  // WHCN -> CWHN
    ksize = {dims[2].size(), dims[1].size()};
    stride = {dims[2].stride(), dims[1].stride()};
    // in_perm = {2, 3, 1, 0};   // NCWH -> WHCN
    // out_perm = {3, 2, 0, 1};  // WHCN -> NCWH
    // ksize = {dims[1].size(), dims[0].size()};
    // stride = {dims[1].stride(), dims[0].stride()};
  }
  // else if (ss.str() == "2 1 3 0 "){
  //   in_perm = {0, 1, 2, 3};   // WHCN -> WHCN
  //   out_perm = {0, 1, 2, 3};  // WHCN -> WHCN
  //   ksize = {dims[1].size(), dims[0].size()};
  //   stride = {dims[1].stride(), dims[0].stride()};
  // }
  else if (ss.str() == "0 3 2 1 "){
    in_perm = {2, 3, 1, 0};   // WHCN -> WHCN
    out_perm = {3, 2, 0, 1};  // WHCN -> WHCN
    ksize = {dims[1].size(), dims[0].size()};
    stride = {dims[1].stride(), dims[0].stride()};
  }
  else{
    return tensorflow::errors::Unimplemented("Unsupported Layout");
  }

  tim::vx::PadType padtype;
  if (window_util::HasPadding(window)) {
    padtype = tim::vx::PadType::SAME;
  } else {
    padtype = tim::vx::PadType::VALID;
  }

  auto input_tensor = GetEvaluatedTensorFor(hlo->operand(0))[0];
  auto gradient_tensor = GetEvaluatedTensorFor(hlo->operand(1))[0];
  auto init_tensor = GetEvaluatedTensorFor(hlo->operand(2))[0];
  auto output_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  auto input_tensor_shape = input_tensor->GetShape();
  std::vector<uint32_t> input_tensor_tmp_shape(in_perm.size());
  for (uint32_t i = 0; i < in_perm.size(); i++) {
    input_tensor_tmp_shape[i] = input_tensor_shape[in_perm[i]];
  }
  tim::vx::TensorSpec input_tensor_tmp_spec(
      input_tensor->GetDataType(),
      input_tensor_tmp_shape,
      tim::vx::TensorAttribute::TRANSIENT);
  auto input_tensor_tmp = graph_->CreateTensor(input_tensor_tmp_spec);

  auto gradient_tensor_shape = gradient_tensor->GetShape();
  std::vector<uint32_t> gradient_tensor_tmp_shape(in_perm.size());
  for (uint32_t i = 0; i < in_perm.size(); i++) {
    gradient_tensor_tmp_shape[i] = gradient_tensor_shape[in_perm[i]];
  }
  tim::vx::TensorSpec gradient_tensor_tmp_spec(
      gradient_tensor->GetDataType(),
      gradient_tensor_tmp_shape,
      tim::vx::TensorAttribute::TRANSIENT);
  auto gradient_tensor_tmp = graph_->CreateTensor(gradient_tensor_tmp_spec);

  auto output_tensor_shape = output_tensor->GetShape();
  std::vector<uint32_t> output_tensor_tmp_shape(out_perm.size());
  for (uint32_t i = 0; i < out_perm.size(); i++) {
    output_tensor_tmp_shape[i] = output_tensor_shape[out_perm[i]];
  }
  tim::vx::TensorSpec output_tensor_tmp_spec(
      output_tensor->GetDataType(),
      output_tensor_tmp_shape,
      tim::vx::TensorAttribute::TRANSIENT);
  auto output_tensor_tmp = graph_->CreateTensor(output_tensor_tmp_spec);
  auto add_init_tensor_tmp = graph_->CreateTensor(output_tensor_tmp_spec);

  auto in_transpose = graph_->CreateOperation<tim::vx::ops::Transpose>(in_perm);
  in_transpose->BindInput(input_tensor).BindOutput(input_tensor_tmp);
  auto source_transpose =
      graph_->CreateOperation<tim::vx::ops::Transpose>(in_perm);
  source_transpose->BindInput(gradient_tensor).BindOutput(gradient_tensor_tmp);

  auto op = graph_->CreateOperation<tim::vx::ops::MaxpoolGrad>(padtype, ksize,
                                                               stride);
  op->BindInputs({input_tensor_tmp, gradient_tensor_tmp})
     .BindOutputs({output_tensor_tmp});

  auto add_init = graph_->CreateOperation<tim::vx::ops::Add>();
  add_init->BindInputs({output_tensor_tmp, init_tensor})
      .BindOutput(add_init_tensor_tmp);

  auto out_transpose =
      graph_->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_transpose->BindInput(add_init_tensor_tmp).BindOutput(output_tensor);

  vsi_run_tensor_container_[hlo].push_back(output_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleConcatenate(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* concat_hlo = Cast<HloConcatenateInstruction>(hlo);
  LOG(INFO) << "HandleConcatenate operand_count: " << hlo->operand_count();
  LOG(INFO) << "HandleConcatenate concatenate_dimension: "
            << concat_hlo->concatenate_dimension();

  if (hlo->operand_count() == 1) {
    auto it = vsi_run_tensor_container_.find(hlo->operand(0));
    vsi_run_tensor_container_[hlo] = it->second;
  } else {
    uint32_t axis =
        shape.dimensions().size() - 1 - concat_hlo->concatenate_dimension();
    auto concat = graph_->CreateOperation<tim::vx::ops::Concat>(
        axis, hlo->operand_count());
    for (int i = 0; i < hlo->operand_count(); i++) {
      const HloInstruction* input = hlo->operand(i);
      auto input_tensor = GetEvaluatedTensorFor(input)[0];
      concat->BindInput(input_tensor);
    }
    auto out_tensor = CreateTensorFromShape(
        shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                : tim::vx::TensorAttribute::TRANSIENT);
    concat->BindOutput(out_tensor);
    vsi_run_tensor_container_[hlo].push_back(out_tensor);
  }

  return Status::OK();
}

Status BaseVisitor::HandleTranspose(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  const HloInstruction* input = hlo->operand(0);
  const Shape& in_shape = input->shape();
  const Shape& out_shape = hlo->shape();

  TF_CHECK_OK(ShapeUtil::ValidateShape(in_shape));
  TF_CHECK_OK(ShapeUtil::ValidateShape(out_shape));
  CHECK(ShapeUtil::SameElementType(in_shape, out_shape));
  CHECK_EQ(hlo->dimensions().size(), in_shape.rank());
  CHECK_EQ(in_shape.rank(), out_shape.rank());

  auto in_tensor = GetEvaluatedTensorFor(input)[0];
  auto out_tensor = CreateTensorFromShape(
      out_shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                  : tim::vx::TensorAttribute::TRANSIENT);

  std::vector<uint32_t> tmpdims;
  auto input_minor_to_major = input->shape().layout().minor_to_major();

  for (auto d : input_minor_to_major) {
    tmpdims.push_back(hlo->dimensions(d));
  }

  std::vector<uint32_t> dims;
  for (auto d : tmpdims) {
    uint32 i = 0;
    for (i = 0; i < input_minor_to_major.size(); i++) {
      if (input_minor_to_major[i] == d) break;
    }
    dims.push_back(i);
  }

  auto transposeOp = graph_->CreateOperation<tim::vx::ops::Transpose>(dims);
  transposeOp->BindInput(in_tensor).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleReverse(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  const HloInstruction* input = hlo->operand(0);
  const Shape& in_shape = input->shape();
  const Shape& out_shape = hlo->shape();

  TF_CHECK_OK(ShapeUtil::ValidateShape(in_shape));
  TF_CHECK_OK(ShapeUtil::ValidateShape(out_shape));
  CHECK(ShapeUtil::SameElementType(in_shape, out_shape));
  // CHECK_EQ(hlo->dimensions().size(), in_shape.rank());
  CHECK_EQ(in_shape.rank(), out_shape.rank());

  auto in_tensor = GetEvaluatedTensorFor(input)[0];
  auto out_tensor = CreateTensorFromShape(
      out_shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                  : tim::vx::TensorAttribute::TRANSIENT);

  std::vector<uint32_t> tmpdims;
  auto input_minor_to_major = input->shape().layout().minor_to_major();

  {
    std::ostringstream ss;
    auto dims = input->shape().dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " input->shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = hlo->shape().dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " hlo->shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < input_minor_to_major.size(); i++) {
      ss << input_minor_to_major[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " input_minor_to_major: " << ss.str();
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < hlo->dimensions().size(); i++) {
      ss << hlo->dimensions(i) << " ";
    }
    LOG(INFO) << __FUNCTION__ << " hlo->dimensions: " << ss.str();
  }

  tmpdims = convert_array<std::vector<uint32_t>>(hlo->dimensions());

  {
    std::ostringstream ss;
    for (int i = 0; i < tmpdims.size(); i++) {
      ss << tmpdims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " tmpdims: " << ss.str();
  }

  std::vector<int32_t> dims;
  for (auto d : tmpdims) {
    uint32 i = 0;
    for (i = 0; i < input_minor_to_major.size(); i++) {
      if (input_minor_to_major[i] == d) {
        dims.push_back(i);
      }
    }
    // dims.push_back(d);
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " Reverse dims0: " << ss.str();
  }

  // auto dims0 = hlo->dimensions();
  // std::vector<int32_t> dims = convert_array<std::vector<int32_t>>(dims0);
  std::reverse(dims.begin(), dims.end());

  {
    std::ostringstream ss;
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " Reverse dims1: " << ss.str();
  }

  auto reverseOp = graph_->CreateOperation<tim::vx::ops::Reverse>(dims);
  reverseOp->BindInput(in_tensor).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleReshape(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  const HloInstruction* input = hlo->operand(0);
  auto in_tensor = GetEvaluatedTensorFor(input)[0];
  std::vector<uint32_t> dims;
  // std::vector<uint32_t> dims =
  //     convert_array<std::vector<uint32_t>>(shape.dimensions());
  LOG(INFO) << __FUNCTION__ << " CCC: " << shape.dimensions().size();

  if (shape.dimensions().size() != 0 && shape.dimensions(0) == 0) {
    auto it = vsi_run_tensor_container_.find(input);
    vsi_run_tensor_container_[hlo].push_back(it->second[0]);
    return Status::OK();
  }

  // {
  //   std::ostringstream ss;
  //   auto minor_to_major = shape.layout().minor_to_major();
  //   for (int i = 0; i < minor_to_major.size(); i++) {
  //     ss << minor_to_major[i] << " ";
  //   }
  //   LOG(INFO) << __FUNCTION__ << " input_minor_to_major: " << ss.str();
  // }

  // {
  //   std::ostringstream ss;
  //   for (int i = 0; i < dims.size(); i++) {
  //     ss << dims[i] << " ";
  //   }
  //   LOG(INFO) << __FUNCTION__ << " CCC dims0: " << ss.str();
  // }

  for (auto d : shape.layout().minor_to_major()) {
    dims.push_back(shape.dimensions(d));
  }

  if (dims.size() == 0) {
    dims.push_back(1);
  }

  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  auto reshapeOp = graph_->CreateOperation<tim::vx::ops::Reshape>(dims);
  reshapeOp->BindInput(in_tensor).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleSlice(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* slice_hlo = Cast<HloSliceInstruction>(hlo);
  const HloInstruction* input = hlo->operand(0);
  auto in_tensor = GetEvaluatedTensorFor(input)[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);
  {
    std::ostringstream ss;
    auto shape = in_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " input shape in TIM-VX: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto shape = out_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " output shape in TIM-VX: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto input_minor_to_major = input->shape().layout().minor_to_major();
    for (int i = 0; i < input_minor_to_major.size(); i++) {
      ss << input_minor_to_major[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " input_minor_to_major: " << ss.str();
  }

  std::vector<int32_t> start(slice_hlo->slice_starts().begin(),
                             slice_hlo->slice_starts().end());
  std::vector<int32_t> limits(slice_hlo->slice_limits().begin(),
                              slice_hlo->slice_limits().end());
  std::vector<int32_t> step(slice_hlo->slice_strides().begin(),
                            slice_hlo->slice_strides().end());

  std::vector<int32_t> length(start.size());
  for (int i = 0; i < start.size(); i++) {
    length[i] = ((limits[i] - start[i]) / step[i]);
  }
  std::reverse(start.begin(), start.end());
  {
    std::ostringstream ss;
    for (int i = 0; i < start.size(); i++) {
      ss << start[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " Reversed start: " << ss.str();
  }

  std::reverse(step.begin(), step.end());
  {
    std::ostringstream ss;
    for (int i = 0; i < step.size(); i++) {
      ss << step[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " Reversed step: " << ss.str();
  }

  std::reverse(length.begin(), length.end());
  {
    std::ostringstream ss;
    for (int i = 0; i < length.size(); i++) {
      ss << length[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " Reversed length: " << ss.str();
  }

  auto sliceOp =
      graph_->CreateOperation<tim::vx::ops::Slice>(0, start, length, step);

  sliceOp->BindInput(in_tensor).BindOutput(out_tensor);
  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleGather(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* gather_hlo = Cast<HloGatherInstruction>(hlo);
  auto params_tensor = GetEvaluatedTensorFor(hlo->operand(0))[0];
  auto indices_tensor = GetEvaluatedTensorFor(hlo->operand(1))[0];

  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);
  {
    std::ostringstream ss;
    auto shape = params_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " params shape in TIM-VX: " << ss.str();
  }
  {
    std::ostringstream ss;
    auto shape = indices_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " indices shape in TIM-VX: " << ss.str();
  }
  {
    std::ostringstream ss;
    auto shape = out_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " output shape in TIM-VX: " << ss.str();
  }

  int axis = gather_hlo->gather_dimension_numbers().start_index_map()[0];
  { LOG(INFO) << __FUNCTION__ << " hlo axis: " << axis; }
  axis = params_tensor->GetShape().size() - 1 - axis;
  { LOG(INFO) << __FUNCTION__ << " axis in TIM-VX: " << axis; }
  auto gatherOp = graph_->CreateOperation<tim::vx::ops::Gather>(axis);
  gatherOp->BindInputs({params_tensor, indices_tensor}).BindOutput(out_tensor);
  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleIota(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* iota_hlo = Cast<HloIotaInstruction>(hlo);
  int64_t iota_dimension = iota_hlo->iota_dimension();
  { LOG(INFO) << __FUNCTION__ << " iota_dimension = " << iota_dimension; }
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  std::vector<uint32_t> outShape(out_tensor->GetShape().begin(),
                                out_tensor->GetShape().end());
  {
    std::ostringstream ss;
    for (int i = 0; i < outShape.size(); i++) {
      ss << outShape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " output shape in TIM-VX: " << ss.str();
  }
  // row-major to column-major
  iota_dimension = outShape.size() - 1 - iota_dimension;
  std::vector<int32_t> dims = {iota_dimension};
  { LOG(INFO) << __FUNCTION__ << " dims in TIM-VX: " << iota_dimension; }
  auto op = graph_->CreateOperation<tim::vx::ops::Broadcast>(outShape, dims);
  if (shape.element_type() == S32) {
    std::vector<int32_t> arrayData;
    for (int i = 0; i < outShape[iota_dimension]; i++) {
      arrayData.push_back(i);
    }
    tim::vx::ShapeType arrayShape = {arrayData.size()};
    tim::vx::TensorSpec arraySpec(tim::vx::DataType::INT32, arrayShape,
                                  tim::vx::TensorAttribute::INPUT);
    auto array_tensor = graph_->CreateTensor(arraySpec, arrayData.data());

    op->BindInput(array_tensor).BindOutput(out_tensor);
  } else if (shape.element_type() == F32) {
    std::vector<float> arrayData;
    for (int i = 0; i < outShape[iota_dimension]; i++) {
      arrayData.push_back(static_cast<float>(i));
      }
    tim::vx::ShapeType arrayShape = {arrayData.size()};
    tim::vx::TensorSpec arraySpec(tim::vx::DataType::FLOAT32, arrayShape,
                                  tim::vx::TensorAttribute::INPUT);
  auto array_tensor = graph_->CreateTensor(arraySpec, arrayData.data());

    op->BindInput(array_tensor).BindOutput(out_tensor);
  } else {
    LOG(FATAL) << "PROCESS " << __FUNCTION__ << " Unsupported Element Type:"
               << static_cast<int>(shape.element_type());
}
  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleDot(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto shape = hlo->shape();
  auto* dot_hlo = Cast<HloDotInstruction>(hlo);
  auto dim_nums = dot_hlo->dot_dimension_numbers();
  const HloInstruction* lhs = hlo->operand(0);
  const HloInstruction* rhs = hlo->operand(1);

  auto lhs_tensor = GetEvaluatedTensorFor(lhs)[0];
  auto rhs_tensor = GetEvaluatedTensorFor(rhs)[0];
  auto out_tensor = CreateTensorFromShape(
      shape, IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                              : tim::vx::TensorAttribute::TRANSIENT);

  {
    std::ostringstream ss;
    const Shape& lhs_shape = lhs->shape();
    auto dims = lhs_shape.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " lhs_shape shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    const Shape& rhs_shape = rhs->shape();
    auto dims = rhs_shape.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " rhs_shape shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < dim_nums.lhs_contracting_dimensions_size(); i++) {
      ss << dim_nums.lhs_contracting_dimensions(i) << " ";
    }
    LOG(INFO) << __FUNCTION__
              << " dot_dimension_numbers lhs_contracting_dimensions: "
              << ss.str();
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < dim_nums.rhs_contracting_dimensions_size(); i++) {
      ss << dim_nums.rhs_contracting_dimensions(i) << " ";
    }
    LOG(INFO) << __FUNCTION__
              << " dot_dimension_numbers rhs_contracting_dimensions: "
              << ss.str();
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < dim_nums.lhs_batch_dimensions_size(); i++) {
      ss << dim_nums.lhs_batch_dimensions(i) << " ";
    }
    LOG(INFO) << __FUNCTION__
              << " dot_dimension_numbers lhs_batch_dimensions: " << ss.str();
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < dim_nums.rhs_batch_dimensions_size(); i++) {
      ss << dim_nums.rhs_batch_dimensions(i) << " ";
    }
    LOG(INFO) << __FUNCTION__
              << " dot_dimension_numbers rhs_batch_dimensions: " << ss.str();
  }

  if (dim_nums.lhs_contracting_dimensions_size() != 1 ||
      dim_nums.rhs_contracting_dimensions_size() != 1 ||
      dim_nums.lhs_batch_dimensions_size() != 0 ||
      dim_nums.rhs_batch_dimensions_size() != 0) {
    return tensorflow::errors::Unimplemented(
        "Only support lhs_contracting_dimensions_size==1 && "
        "rhs_contracting_dimensions_size==1"
        " && lhs_batch_dimensions_size==0 && rhs_batch_dimensions_size==0");
  }

  bool transpose_a, transpose_b;
  if (dim_nums.lhs_contracting_dimensions(0) == 1) {
    transpose_a = false;
  } else {
    transpose_a = true;
  }

  if (dim_nums.rhs_contracting_dimensions(0) == 1) {
    transpose_b = true;
  } else {
    transpose_b = false;
  }
  LOG(INFO) << __FUNCTION__ << " transpose_a: " << transpose_a;
  LOG(INFO) << __FUNCTION__ << " transpose_b: " << transpose_b;

  auto matmul = graph_->CreateOperation<tim::vx::ops::Matmul>(
      transpose_a, transpose_b, false, false);
  matmul->BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::HandleParameter(HloInstruction* hlo) {
  CHECK_LT(hlo->parameter_number(), arg_literals_.size());
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto& input_literal = arg_literals_[hlo->parameter_number()];
  VLOG(2) << "Parameter evaluated to: " << input_literal.ToString();
  DCHECK(Shape::Equal().MinorToMajorOnlyInLayout()(hlo->shape(),
                                                   input_literal.shape()))
      << "parameter shape is: "
      << ShapeUtil::HumanStringWithLayout(hlo->shape())
      << ", but input literal shape is: "
      << ShapeUtil::HumanStringWithLayout(input_literal.shape());

  if (vsi_run_tensor_container_.find(hlo) == vsi_run_tensor_container_.end()) {
    ShapeIndex shapeIndex({});
    void* buffer = input_literal.untyped_data(shapeIndex);
    auto timTensor = CreateTensorFromShape(input_literal.shape());
    timTensor->CopyDataToTensor(buffer);
    vsi_run_tensor_container_[hlo].push_back(timTensor);
    kVsiInputId_[hlo->parameter_number()] = timTensor->GetId();
  }
  LOG(INFO) << "kVsiInputId_: " << kVsiInputId_.size();
  return Status::OK();
}

Status BaseVisitor::HandleConstant(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS Constant";

  if (vsi_run_tensor_container_.find(hlo) == vsi_run_tensor_container_.end()) {
    ShapeIndex shapeIndex({});

    auto& literal = hlo->literal();
    const void* buffer = literal.untyped_data(shapeIndex);
    auto timTensor = CreateTensorFromShape(literal.shape());
    timTensor->CopyDataToTensor(buffer);
    vsi_run_tensor_container_[hlo].push_back(timTensor);
  }

  return Status::OK();
}

Status BaseVisitor::HandleConvolution(HloInstruction* hlo) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;

  auto lhs = hlo->operand(0);
  auto rhs = hlo->operand(1);
  const auto& window = hlo->window();
  const Shape& result_shape = hlo->shape();
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();

  TF_CHECK_OK(ShapeUtil::ValidateShape(lhs_shape));
  TF_CHECK_OK(ShapeUtil::ValidateShape(rhs_shape));
  CHECK(lhs_shape.IsArray());
  CHECK(rhs_shape.IsArray());
  CHECK(ShapeUtil::SameElementType(lhs_shape, rhs_shape));
  CHECK(ShapeUtil::SameElementType(lhs_shape, result_shape));

  const auto& dnums = hlo->convolution_dimension_numbers();
  const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
  CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
  CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
  CHECK_EQ(num_spatial_dims, 2); /*vsi requirement*/
  CHECK_GE(num_spatial_dims, 0);
  CHECK_EQ(window.dimensions_size(), num_spatial_dims);

  const auto lhs_rank = lhs_shape.rank();
  const auto rhs_rank = rhs_shape.rank();
  CHECK_EQ(num_spatial_dims + 2, lhs_rank);
  CHECK_EQ(num_spatial_dims + 2, rhs_rank);

  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferConvolveShape(
          lhs_shape, rhs_shape, hlo->feature_group_count(),
          hlo->batch_group_count(), window, dnums, absl::nullopt));
  CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  // prepare parameter for vsi.
  std::vector<uint32_t> input_dim;
  input_dim.push_back(dnums.input_batch_dimension());
  input_dim.push_back(dnums.input_feature_dimension());

  std::vector<uint32_t> weight_dim;
  weight_dim.push_back(dnums.kernel_output_feature_dimension());
  weight_dim.push_back(dnums.kernel_input_feature_dimension());
  for (size_t i = 2; i < lhs_rank; i++) {
    input_dim.push_back(dnums.input_spatial_dimensions(i - 2));
    weight_dim.push_back(dnums.kernel_spatial_dimensions(i - 2));
  }

  LOG(INFO) << "dnums.kernel_output_feature_dimension: "
            << dnums.kernel_output_feature_dimension();
  LOG(INFO) << "dnums.kernel_input_feature_dimension: "
            << dnums.kernel_input_feature_dimension();
  LOG(INFO)
      << "rhs_shape.dimensions()[dnums.kernel_output_feature_dimension()]: "
      << rhs_shape.dimensions()[dnums.kernel_output_feature_dimension()];

    {
    std::ostringstream ss;
    auto dims = lhs_shape.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " lhs_shape shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = rhs_shape.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " rhs_shape shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = result_shape.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " result_shape shape: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].window_dilation() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " window_dilation: " << ss.str();
  }

  {
    std::ostringstream ss;
    auto dims = window.dimensions();
    for (int i = 0; i < dims.size(); i++) {
      ss << dims[i].base_dilation() << " ";
    }
    LOG(INFO) << __FUNCTION__ << " base_dilation: " << ss.str();
  }

/*prepare input and weight: change layout to WHCN, layout minor to
   * major:{0,1,2,3}*/
  auto input = InsertTranspose(lhs, input_dim);
  auto weight = InsertTranspose(rhs, weight_dim);

  std::array<uint32_t, 2> ksize = {window.dimensions(1).size(),
                                   window.dimensions(0).size()};

  std::array<int32_t, 4> xla_pad = {
      window.dimensions(1).padding_low(),
      window.dimensions(1).padding_high(),  // top bottom
      window.dimensions(0).padding_low(),
      window.dimensions(0).padding_high()};  // left, right

  {
    std::ostringstream ss;
    for (int i = 0; i < xla_pad.size(); i++) {
      ss << xla_pad[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " xla_pad: " << ss.str();
  }

  std::shared_ptr<tim::vx::Operation> convOp;
  bool need_slice = false;

  // conv2d don't support base dilation
  if (window.dimensions(1).base_dilation() != 1 or
      window.dimensions(0).base_dilation() != 1) {
    // forward stride equal to backward base dilation
    std::array<uint32_t, 2> forward_stride = {
        window.dimensions(1).base_dilation(),
        window.dimensions(0).base_dilation()};
    std::array<uint32_t, 2> output_padding = {0, 0};
    std::array<uint32_t, 4> forward_pad = {0, 0, 0, 0};
    for (int i = 0, j = 0; i < 4; i++) {
      int real_forward_pad = ksize[i % 2] - xla_pad[i] - 1;
      if (real_forward_pad < 0) {
        // DeConv2d parameter pad not support negative pad,
        // so we implement it by do output padding.
        output_padding[j++] = - real_forward_pad;
      } else {
        forward_pad[i] = real_forward_pad;
      }
    }
    // deconv2d need the params in forword, but not backword
    convOp = graph_->CreateOperation<tim::vx::ops::DeConv2d>(
        rhs_shape.dimensions()[dnums.kernel_output_feature_dimension()],
        tim::vx::PadType::AUTO, ksize, forward_stride, output_padding,
        forward_pad);
    {
      std::ostringstream ss;
      for (int i = 0; i < ksize.size(); i++) {
      ss << ksize[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " DeConv2d param ksize: " << ss.str();
    }
    {
      std::ostringstream ss;
      for (int i = 0; i < forward_stride.size(); i++) {
        ss << forward_stride[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " DeConv2d param forward_stride: " << ss.str();
    }
    {
      std::ostringstream ss;
      for (int i = 0; i < output_padding.size(); i++) {
        ss << output_padding[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " DeConv2d param output_padding: " << ss.str();
    }
    {
      std::ostringstream ss;
      for (int i = 0; i < forward_pad.size(); i++) {
        ss << forward_pad[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " DeConv2d param forward_pad: " << ss.str();
    }
    // Deconv2d will reverse the weight, but the weight already reversed.
    // So we must reverse the weight again.
    std::vector<int32_t> dims = {1, 0};
    auto reverseOp = graph_->CreateOperation<tim::vx::ops::Reverse>(dims);
    auto weight_spec = weight->GetSpec();
    auto weight_tmp = graph_->CreateTensor(weight_spec);
    reverseOp->BindInput(weight).BindOutput(weight_tmp);
    weight = weight_tmp;
  } else {
    std::array<uint32_t, 2> stride = {window.dimensions(1).stride(),
                                      window.dimensions(0).stride()};
    std::array<uint32_t, 2> dilation = {window.dimensions(1).window_dilation(),
                                        window.dimensions(0).window_dilation()};
    std::array<uint32_t, 4> conv_pad = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
      if (xla_pad[i] < 0) {
        need_slice = true;
      } else {
        conv_pad[i] = (uint32_t)(xla_pad[i]);
      }
    }

    convOp = graph_->CreateOperation<tim::vx::ops::Conv2d>(
        rhs_shape.dimensions()[dnums.kernel_output_feature_dimension()],
        tim::vx::PadType::AUTO, ksize, stride, dilation, conv_pad);
    {
      std::ostringstream ss;
      for (int i = 0; i < ksize.size(); i++) {
        ss << ksize[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " Conv2d param ksize: " << ss.str();
}
    {
      std::ostringstream ss;
      for (int i = 0; i < stride.size(); i++) {
        ss << stride[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " Conv2d param stride: " << ss.str();
    }
    {
      std::ostringstream ss;
      for (int i = 0; i < dilation.size(); i++) {
        ss << dilation[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " Conv2d param dilation: " << ss.str();
    }
    {
      std::ostringstream ss;
      for (int i = 0; i < conv_pad.size(); i++) {
        ss << conv_pad[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " Conv2d param conv_pad: " << ss.str();
    }
  }

  LOG(INFO) << __FUNCTION__
            << " batch_dimension: " << dnums.input_batch_dimension() << " "
            << dnums.output_batch_dimension();

  std::vector<uint32_t> perm;

  auto out_tensor = CreateTensorFromShape(
      hlo->shape(), IsRootHlo(hlo) ? tim::vx::TensorAttribute::OUTPUT
                                     : tim::vx::TensorAttribute::TRANSIENT);
  auto out_tensor_spec = out_tensor->GetSpec();
  auto out_tensor_shape = out_tensor_spec.GetShapeType();
  std::vector<uint32_t> out_tensor_tmp_shape;
  // In HloConvolution:
  // lhs layout is [batch, z/depth/features, spatial_dims], also known as NCHW.
  // rhs layout is [output-z, input-z, spatial_dims], also known as OIHW.
  // output layout is [batch, z, spatial_dims], it is as same as lhs layout.
  //
  // For example: the first Conv2D in lenet.
  // Conv2D:
  // lhs shape: [8, 1, 28, 28]
  // rhs shape: [6, 1, 5, 5]
  // output shape: [8, 24, 24, 6]
  // batch_dimension are 0, both lhs and output.
  //
  // Conv2DBackpropInput:
  // lhs shape: [8, 6, 24, 24]
  // rhs shape: [1, 6, 5, 5]
  // output shape: [8, 28, 28, 1]
  // batch_dimension are 0, both lhs and output.
  //
  // But when Conv2DBackpropFilter :
  // lhs shape: [1, 8, 28, 28]
  // rhs shape: [6, 8, 24, 24]
  // output shape: [5, 5, 1, 6]
  // lhs batch_dimension is 0, output batch_dimension is 2.

  if (dnums.input_batch_dimension() != dnums.output_batch_dimension()) {
    perm = {2, 3, 0, 1};
    out_tensor_tmp_shape = {out_tensor_shape[2], out_tensor_shape[3],
                            out_tensor_shape[0], out_tensor_shape[1]};
    LOG(INFO) << __FUNCTION__ << " BackpropFilter X";
  } else {
    perm = {2, 0, 1, 3};
    out_tensor_tmp_shape = {out_tensor_shape[1], out_tensor_shape[2],
                            out_tensor_shape[0], out_tensor_shape[3]};
    LOG(INFO) << __FUNCTION__ << " Other Conv X";
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < out_tensor_tmp_shape.size(); i++) {
      ss << out_tensor_tmp_shape[i] << ", ";
    }
    LOG(INFO) << __FUNCTION__ << " out_tensor_tmp_shape: " << ss.str();
  }
  tim::vx::TensorSpec out_tensor_tmp_sec(out_tensor_spec.GetDataType(),
                                         out_tensor_tmp_shape,
                                         tim::vx::TensorAttribute::TRANSIENT);
  auto out_tensor_tmp = graph_->CreateTensor(out_tensor_tmp_sec);

if (need_slice) {
    // Conv2d do not support negative pad, negative pad is equivalent to slice.
    auto input_shape = input->GetShape();        // WHCN
    std::vector<int32_t> start = {0, 0, 0, 0};   // whcn
    std::vector<int32_t> length = {0, 0, 0, 0};  // whcn
    start[0] = abs(xla_pad[0]);                            // top
    start[1] = abs(xla_pad[2]);                            // left
    length[0] = input_shape[0] + xla_pad[0] + xla_pad[1];  // w + left + right
    length[1] = input_shape[1] + xla_pad[2] + xla_pad[3];  // h + top + bottom
    length[2] = input_shape[2];                            // c
    length[3] = input_shape[3];                            // n

    {
      std::ostringstream ss;
      for (int i = 0; i < 4; i++) {
        ss << start[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " slice start: " << ss.str();
    }

    {
      std::ostringstream ss;
      for (int i = 0; i < 4; i++) {
        ss << length[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " slice length: " << ss.str();
    }

    auto sliceOp = graph_->CreateOperation<tim::vx::ops::Slice>(0, start, length);

    std::vector<uint32_t> slice_shape = {
        (uint32_t)(length[3]), (uint32_t)(length[2]), (uint32_t)(length[1]),
        (uint32_t)(length[0])};  // nchw

    tim::vx::TensorSpec slice_tensor_spec(out_tensor_spec.GetDataType(),
                                          slice_shape,
                                          tim::vx::TensorAttribute::TRANSIENT);

    auto slice_tensor = graph_->CreateTensor(slice_tensor_spec);

    sliceOp->BindInput(input).BindOutput(slice_tensor);
    convOp->BindInput(slice_tensor)
        .BindInput(weight)
        .BindOutput(out_tensor_tmp);

  } else {
  convOp->BindInput(input).BindInput(weight).BindOutput(out_tensor_tmp);
}

  auto transposeOp = graph_->CreateOperation<tim::vx::ops::Transpose>(perm);
  transposeOp->BindInput(out_tensor_tmp).BindOutput(out_tensor);

  vsi_run_tensor_container_[hlo].push_back(out_tensor);
  return Status::OK();
}

#if 1
Status BaseVisitor::CustomCallLogSoftmax(HloInstruction* inst,
                                         BaseVisitor* bv) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto in_tensor = bv->GetEvaluatedTensorFor(inst->operand(0))[0];
  auto shape = inst->shape();
  auto out_tensor = bv->CreateTensorFromShape(
      shape, IsRootHlo(inst) ? tim::vx::TensorAttribute::OUTPUT
                               : tim::vx::TensorAttribute::TRANSIENT);
  std::string paramters = Cast<HloCustomCallInstruction>(inst)->opaque();

  auto op = bv->graph_->CreateOperation<tim::vx::ops::LogSoftmax>(0);

  op->BindInput(in_tensor).BindOutput(out_tensor);

  bv->vsi_run_tensor_container_[inst].push_back(out_tensor);
  return Status::OK();
}

Status BaseVisitor::CustomCallTopK(HloInstruction* inst, BaseVisitor* bv) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto in_tensor = bv->GetEvaluatedTensorFor(inst->operand(0))[0];
  auto tuple_shape = inst->shape();
  auto value_shape = *(tuple_shape.mutable_tuple_shapes(0));
  auto indices_shape = *(tuple_shape.mutable_tuple_shapes(1));

  auto value_tensor = bv->CreateTensorFromShape(
      value_shape, IsRootHlo(inst) ? tim::vx::TensorAttribute::OUTPUT
                                     : tim::vx::TensorAttribute::TRANSIENT);
  auto indices_tensor = bv->CreateTensorFromShape(
      indices_shape, IsRootHlo(inst) ? tim::vx::TensorAttribute::OUTPUT
                                       : tim::vx::TensorAttribute::TRANSIENT);

  xla::vsiplugin::AttributeMap attribute_map(
      Cast<HloCustomCallInstruction>(inst)->opaque());
  int32_t k = *(attribute_map.GetAttributeAsInt("k"));

  auto op = bv->graph_->CreateOperation<tim::vx::ops::Topk>(k);

  op->BindInput(in_tensor).BindOutputs({value_tensor, indices_tensor});

  bv->vsi_run_tensor_container_[inst].push_back(value_tensor);
  bv->vsi_run_tensor_container_[inst].push_back(indices_tensor);
  return Status::OK();
}

/* static */ Status BaseVisitor::CustomCallSqueeze(HloInstruction* inst,
                                                   BaseVisitor* bv) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;
  auto in_tensor = bv->GetEvaluatedTensorFor(inst->operand(0))[0];
  auto shape = inst->shape();
  auto out_tensor = bv->CreateTensorFromShape(
      shape, IsRootHlo(inst) ? tim::vx::TensorAttribute::OUTPUT
                               : tim::vx::TensorAttribute::TRANSIENT);
  {
    std::ostringstream ss;
    auto shape = in_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " input shape in TIM-VX: " << ss.str();
  }
  {
    std::ostringstream ss;
    auto shape = out_tensor->GetShape();
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " output shape in TIM-VX: " << ss.str();
  }
  xla::vsiplugin::AttributeMap attribute_map(
      Cast<HloCustomCallInstruction>(inst)->opaque());

  auto axis_ =
      attribute_map.GetAttributeInt64Vector("squeeze_dims").ValueOrDie();
  std::vector<uint32_t> axis;
  int dims = in_tensor->GetShape().size();
  for (auto it = axis_.rbegin(); it != axis_.rend(); it++) {
    axis.push_back(dims - 1 - *it);
  }

  {
    std::ostringstream ss;
    for (int i = 0; i < axis.size(); i++) {
      ss << axis[i] << " ";
    }
    LOG(INFO) << __FUNCTION__ << " axis in TIM-VX: " << ss.str();
  }
  auto op = bv->graph_->CreateOperation<tim::vx::ops::Squeeze>(axis);
  op->BindInput(in_tensor).BindOutput(out_tensor);
  bv->vsi_run_tensor_container_[inst].push_back(out_tensor);
  return Status::OK();
}

// Status BaseVisitor::CustomCallSoftmax(HloInstruction* inst,
//                                          BaseVisitor* bv) {
//   LOG(INFO) << "PROCESS " << __FUNCTION__;
//   LOG(INFO) << "PROTO " << inst->ToString();
//   auto in_tensor = bv->GetEvaluatedTensorFor(inst->operand(0))[0];
//   auto shape = inst->shape();
//   auto out_tensor = bv->CreateTensorFromShape(
//       shape, IsRootHlo(inst) ? tim::vx::TensorAttribute::OUTPUT
//                                : tim::vx::TensorAttribute::TRANSIENT);
//   std::vector<std::shared_ptr<tim::vx::Tensor>> in_tensors;
//   std::vector<std::shared_ptr<tim::vx::Tensor>> out_tensors;
//   in_tensors.push_back(in_tensor);
//   out_tensors.push_back(out_tensor);
//   auto status = DoVisitor::CustomCallSoftmax(
//       inst, in_tensors, out_tensors,
//       bv->vsi_run_tensor_container_, bv->graph_);
//   return status;
// }

// Status BaseVisitor::CustomCallArgMax(HloInstruction* inst,
//                                          BaseVisitor* bv) {
//   LOG(INFO) << "PROCESS " << __FUNCTION__;
//   LOG(INFO) << "PROTO " << inst->ToString();
//   auto in_tensor = bv->GetEvaluatedTensorFor(inst->operand(0))[0];
//   auto shape = inst->shape();
//   auto out_tensor = bv->CreateTensorFromShape(
//       shape, IsRootHlo(inst) ? tim::vx::TensorAttribute::OUTPUT
//                                : tim::vx::TensorAttribute::TRANSIENT);

//   xla::vsiplugin::AttributeMap attribute_map(
//       Cast<HloCustomCallInstruction>(inst)->opaque());
//   int32_t dim = *(attribute_map.GetAttributeAsInt("dimension"));
//   std::vector<std::shared_ptr<tim::vx::Tensor>> in_tensors;
//   std::vector<std::shared_ptr<tim::vx::Tensor>> out_tensors;
//   in_tensors.push_back(in_tensor);
//   out_tensors.push_back(out_tensor);
//   auto status = DoVisitor::CustomCallArgmax(
//       inst, in_tensors, out_tensors, dim,
//       bv->vsi_run_tensor_container_, bv->graph_);
//   return status;
// }

Status BaseVisitor::HandleCustomCall(HloInstruction* inst) {
  LOG(INFO) << "PROCESS " << __FUNCTION__;

  auto* cc_hlo = Cast<HloCustomCallInstruction>(inst);
  auto cc_name = cc_hlo->custom_call_target();

  std::string attr = cc_hlo->opaque();
  LOG(INFO) << __FUNCTION__ << "Serialised attrs are: " << attr;

  if (custom_call_map_.count(cc_name)) {
    return custom_call_map_[cc_name](inst, this);
  }
  LOG(FATAL) << "no target for custom call: " << cc_name;
  return Status::OK();
}
#endif
}  // namespace vsiplugin
}  // namespace xla
