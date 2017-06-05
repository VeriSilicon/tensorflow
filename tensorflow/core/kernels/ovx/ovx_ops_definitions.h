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

// HVX internal supported ops names
enum class SupportedOpType {
  BIASADD_F,
  CONV2D_F,
  CONVOLUTION_RELUE,
  CONVOLUTION_RELUE_POOL,
  FULLCONNECT,
  FullConnectRelu,
  SOFTMAX_F,
  MATMUL_F,
  RELU_F,
  RELUX_F,
  AVGPOOL_F,
  MAXPOOL_F,
  RESHAPE,
  FLATTEN,
  TANH_F,
  SIGMOID_F,
  CONCAT_F,
  LRN_F,
  INPUT,
  OUTPUT,
  MIN_F,
  MIN_F_REF,
  MAX_F,
  MAX_F_REF,
  LRN,
  CONCAT,
  IDENTITY,
  NOP,
  OP_CONST, /* OP_ is required to avoid compilation error on windows */
  CHECK,
  CLOSE_FLOAT32,
  CLOSE_QINT8,
  CLOSE_Q_QINT8,
  CLOSE_INT32,
  CLOSE_QINT32,
  PPRINT_8,
  PPRINT_32,
  PPRINT_FLOAT,
  PREFREE,
  QUANTIZEDCONV2D_8X8TO32,
  QUANTIZEDCONV2D_8X8TO32_REF,
  QUANTIZEDMATMUL_8X8TO32,
  QUANTIZEDMATMUL_8X8TO32_REF,
  QUANTIZEDOWNANDSHRINKRANGE_32TO8,
  QUANTIZEDOWNANDSHRINKRANGE_32TO8_REF,
  QUANTIZEDRELU_8,
  QUANTIZEDRELU_8_REF,
  QUANTIZEDRELUX_8,
  QUANTIZEDRELUX_8_REF,
  QUANTIZEDMAXPOOL_8,
  QUANTIZEDMAXPOOL_8_REF,
  QUANTIZEDAVGPOOL_8,
  QUANTIZEDAVGPOOL_8_REF,
  QUANTIZEDCONCAT_8,
  QUANTIZEDCONCAT_8_REF,
  QUANTIZEDBIASADD_8P8TO32,
  QUANTIZEDBIASADD_8P8TO32_REF,
  QUANTIZED_RESHAPE,
  QUANTIZE,
  QUANTIZE_REF,
  DEQUANTIZE,
  DEQUANTIZE_REF,
  SUPERNODE_8X8P8TO8,
  SUPERNODE_8X8P8TO8_REF,
  QUANTIZEDFLATTEN,
  VARIABLE,
  ASSIGN,
  SUPPORTED_OP_TYPE_COUNT,
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
