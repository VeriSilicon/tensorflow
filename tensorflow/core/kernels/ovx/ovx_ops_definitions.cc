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
  {"Input", SupportedOpType::INPUT},
  {"Output", SupportedOpType::OUTPUT},
  {"NoOp", SupportedOpType::NOP},
  // Tensorflow op name
  {"Min", SupportedOpType::MIN},
  {"Max", SupportedOpType::MAX},
  {"Softmax", SupportedOpType::SOFTMAX},
  {"Placeholder", SupportedOpType::INPUT},
  {"Conv2D", SupportedOpType::CONV2D},
  {"Add", SupportedOpType::ADD},
  {"BiasAdd", SupportedOpType::ADD},
  {"Relu", SupportedOpType::RELU},
  {"Pool", SupportedOpType::POOL},
  {"ConvolutionReluPool", SupportedOpType::CONVOLUTION_RELU_POOL},
  {"ConvolutionRelu", SupportedOpType::CONVOLUTION_RELU},
  {"FullConnectRelu", SupportedOpType::FULLCONNECT_RELU},
  {"FullConnect", SupportedOpType::FULLCONNECT},
  {"MatMul", SupportedOpType::MATMUL},
  {"Reshape", SupportedOpType::RESHAPE},
  {"LRN", SupportedOpType::LRN},
  {"Concat", SupportedOpType::CONCAT},
  {"Identity", SupportedOpType::NOP},
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
