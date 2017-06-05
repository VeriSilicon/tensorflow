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

#include "tensorflow/core/kernels/ovx/ovx_ops_definitions.h"

#include <unordered_map>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {

const std::unordered_map<string, SupportedOpType> OP_NAME_TO_SOC_OP_TYPE_MAP{
  // Custom Op name
  {"INPUT", SupportedOpType::INPUT},
  {"OUTPUT", SupportedOpType::OUTPUT},
  {"NoOp", SupportedOpType::NOP},
  {IGraphTransferOpsDefinitions::FLATTEN_OP_NAME, SupportedOpType::FLATTEN},
  // Tensorflow op name
  {"Min", SupportedOpType::MIN_F},
  {"Max", SupportedOpType::MAX_F},
  {"QuantizeV2", SupportedOpType::QUANTIZE},
  {"Dequantize", SupportedOpType::DEQUANTIZE},
  {"Softmax", SupportedOpType::SOFTMAX_F},
  {"Placeholder", SupportedOpType::NOP},
  {"Conv2D", SupportedOpType::CONV2D_F},
  {"Add", SupportedOpType::BIASADD_F},
  {"BiasAdd", SupportedOpType::BIASADD_F},
  {"Relu", SupportedOpType::RELU_F},
  {"MaxPool", SupportedOpType::MAXPOOL_F},
  {"AvgPool", SupportedOpType::AVGPOOL_F},
  {"ConvolutionReluePool", SupportedOpType::CONVOLUTION_RELUE_POOL},
  {"ConvolutionRelu", SupportedOpType::CONVOLUTION_RELUE},
  {"FullConnectRelu", SupportedOpType::FullConnectRelu},
  {"FullConnect", SupportedOpType::FULLCONNECT},
  {"MatMul", SupportedOpType::MATMUL_F},
  {"Reshape", SupportedOpType::RESHAPE},
  {"LRN", SupportedOpType::LRN},
  {"Concat", SupportedOpType::CONCAT},
  {"Identity", SupportedOpType::IDENTITY},
  {"QuantizedConv2D", SupportedOpType::QUANTIZEDCONV2D_8X8TO32},
  {"QuantizedMatMul", SupportedOpType::QUANTIZEDMATMUL_8X8TO32},
  {"QuantizeDownAndShrinkRange",
   SupportedOpType::QUANTIZEDOWNANDSHRINKRANGE_32TO8},
  {"QuantizedRelu", SupportedOpType::QUANTIZEDRELU_8},
  {"QuantizedReluX", SupportedOpType::QUANTIZEDRELUX_8},
  {"QuantizedMaxPool", SupportedOpType::QUANTIZEDMAXPOOL_8},
  {"QuantizedAvgPool", SupportedOpType::QUANTIZEDAVGPOOL_8},
  {"QuantizedConcat", SupportedOpType::QUANTIZEDCONCAT_8},
  {"QuantizedBiasAdd", SupportedOpType::QUANTIZEDBIASADD_8P8TO32},
};

/* static */ const IGraphTransferOpsDefinitions&
OvxOpsDefinitions::getInstance() {
  const static OvxOpsDefinitions instance{};
  return instance;
}

int OvxOpsDefinitions::GetTotalOpsCount() const {
  return static_cast<int>(SupportedOpType::SUPPORTED_OP_TYPE_COUNT);
}

int OvxOpsDefinitions::GetOpIdFor(const string& op_type) const {
  if (OP_NAME_TO_SOC_OP_TYPE_MAP.count(op_type) > 0) {
    return static_cast<int>(OP_NAME_TO_SOC_OP_TYPE_MAP.at(op_type));
  }
  return IGraphTransferOpsDefinitions::INVALID_OP_ID;
}

GraphTransferInfo::Destination OvxOpsDefinitions::GetTransferDestination()
    const {
  return GraphTransferInfo::OVX;
}
};
