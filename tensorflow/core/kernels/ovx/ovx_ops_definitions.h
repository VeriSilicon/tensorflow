/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/ 
#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_OVX_OVX_OPS_DEFINITIONS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_OVX_OVX_OPS_DEFINITIONS_H_

#include "i_graph_transfer_ops_definitions.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

enum class OvxPoolType {
  MAX,
  AVG
};

enum class OvxPaddingType {
  AUTO,
  VALID,
  SAME
};

// HVX internal supported ops names
enum class SupportedOpType {
  ADD,
  CONV2D,
  CONVOLUTION_RELU,
  CONVOLUTION_RELU_POOL,
  FULLCONNECT,
  FULLCONNECT_RELU,
  SOFTMAX,
  POOL,
  LEAKY_RELU,
  LRN,
  CONCAT,
  NOP,

  INPUT,
  OUTPUT,
  RELU,
  MATMUL,
  RESHAPE,

  MIN,
  MAX,

  OP_CONST, /* OP_ is required to avoid compilation error on windows */
  CHECK,
  VARIABLE,
  ASSIGN,
  SUPPORTED_OP_TYPE_COUNT,

  NA = IGraphTransferOpsDefinitions::INVALID_OP_ID,
};

// OvxOpsDefinitions provides ops definitons supported in ovx library
// TODO(satok): add a functionality to call functions in ovx library
class OvxOpsDefinitions final : public IGraphTransferOpsDefinitions {
 public:
  static const IGraphTransferOpsDefinitions& getInstance();

  int GetTotalOpsCount() const final;
  int GetOpIdFor(const string& op_type) const final;
  GraphTransferInfo::Destination GetTransferDestination() const final;

 private:
  OvxOpsDefinitions() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(OvxOpsDefinitions);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_OVX_OVX_OPS_DEFINITIONS_H
