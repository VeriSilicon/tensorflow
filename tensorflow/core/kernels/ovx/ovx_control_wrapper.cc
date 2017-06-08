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

#include "tensorflow/core/kernels/ovx/ovx_control_wrapper.h"

#include "tensorflow/core/kernels/ovx/ovx_ops_definitions.h"

#ifdef USE_OVX_LIBS
//#include "tensorflow/core/platform/ovx/soc_interface.h"
#include "soc_interface.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#endif

namespace tensorflow {

const bool DBG_DUMP_VERIFICATION_STRING = true;
const bool SHOW_DBG_IN_SOC = false;
const bool DBG_DUMP_INPUT_TENSOR_AS_FLOAT_DATA = false;

/* static */ GraphTransferInfo::NodeInfo* OvxControlWrapper::FindNodeInfo(
    const string& name, GraphTransferInfo* graph_transfer_info) {
  for (GraphTransferInfo::NodeInfo& node_info :
       *graph_transfer_info->mutable_node_info()) {
    if (node_info.name() == name) {
      return &node_info;
    }
  }
  return nullptr;
}

#ifdef USE_OVX_LIBS
int OvxControlWrapper::GetVersion() {
  return soc_interface_GetSocControllerVersion();
}

bool OvxControlWrapper::Init(const RemoteFusedGraphExecuteInfo& info) {
  soc_interface_SetLogLevel(SHOW_DBG_IN_SOC ? -1 /* debug */ : 0 /* info */);
  graph_transferer_.SetSerializedGraphTransferInfo(
      info.serialized_executor_parameters());

  graph_transferer_.GraphNodeMerge();

  execute_info_ = &info;
  return soc_interface_Init();
}

bool OvxControlWrapper::Finalize() { return soc_interface_Finalize(); }
bool OvxControlWrapper::SetupGraph() {
  bool ret;

  std::unordered_map<int, uint32> ovxnode_map;
  std::unordered_map<int, std::vector<std::tuple<int, int>>> input_ports_map;
  std::unordered_map<int, std::vector<uint32>> ovxtensor_map;
  std::vector<int> graph_inputs;
  std::vector<int> graph_outputs;
  ret = true;
  // Copy graph transfer info to modify to adapt ovx nn library
  GraphTransferInfo& graph_transfer_info =
      graph_transferer_.GetMutableGraphTransferInfo();

  if (DBG_DUMP_VERIFICATION_STRING) {
    GraphTransferer gt;
    gt.SetSerializedGraphTransferInfo(graph_transfer_info.SerializeAsString());
    gt.DumpVerificationStringOfNodeTransferParams();
  }

  return false;

  for (const GraphTransferInfo::GraphInputNodeInfo& graph_input :
       graph_transfer_info.graph_input_node_info()) {
    GraphTransferInfo::NodeInfo* node = FindNodeInfo(
            graph_input.name(), &graph_transfer_info);
    graph_inputs.push_back(node->node_id());
  }

  for (const GraphTransferInfo::GraphOutputNodeInfo& graph_output :
       graph_transfer_info.graph_output_node_info()) {
    GraphTransferInfo::NodeInfo* node = FindNodeInfo(
            graph_output.name(), &graph_transfer_info);
    graph_outputs.push_back(node->node_id());
  }

  //TODO: Fix me
  //graph_inputs.push_back(0x0c);
  //graph_outputs.push_back(0x1d);

  // Count tensors
  int tensor_count = 0;
  for (const GraphTransferInfo::NodeOutputInfo& output_info :
       graph_transfer_info.node_output_info()) {
    tensor_count += output_info.max_byte_size_size();
    const int node_id = output_info.node_id();
    ovxtensor_map.emplace(node_id, std::vector<uint32>());
  }
  tensor_count += graph_transfer_info.const_node_info_size();

  int node_count = graph_transfer_info.node_info_size() - graph_inputs.size();
  // Instantiate graph
  soc_interface_InstantiateGraph(graph_inputs.size(),
          graph_outputs.size(), tensor_count, node_count);


  // Setup op nodes
  for (const GraphTransferInfo::NodeInfo& params :
       graph_transfer_info.node_info()) {
    const int node_id = params.node_id();
    const int op_id = params.soc_op_id();

    //TODO: if not input
    if (op_id != 21) {
      const uint32 ovxnode_id = soc_interface_AppendNode(
                               params.name().c_str(), op_id);
      ovxnode_map.emplace(node_id, ovxnode_id);
    }
  }

  #define OVX_NODE_ID_NA   ((uint32)-1)
  #define OVX_TENSOR_ID_NA   ((uint32)-1)
  #define OVX_MAX_DIM_NUM    (4)
  // Init graph input tensor
  for (const int node_id : graph_inputs) {
    // TODO: Fix me
    // Find node by id
    // Get node shape
    uint32 shape[] = {1, 28, 28, 1};
    int dim_num = 4;
    int dtype = 0;
    uint32 tensor_id = soc_interface_AppendTensor(
       OVX_NODE_ID_NA, 0, shape, dim_num, nullptr, 0, dtype);
    ovxtensor_map[node_id].push_back(tensor_id);
    soc_interface_SetGraphInputTensor(tensor_id);
  }

  // Init graph output tensor
  for (const int node_id : graph_outputs) {
    // TODO: Fix me
    // Find node by id
    // Get node shape
    uint32 shape[] = {1, 1, 1, 10};
    int dim_num = 4;
    int dtype = 0;
    uint32 ovxnode_id = ovxnode_map[node_id];

    // TODO: Fix output port
    uint32 tensor_id = soc_interface_AppendTensor(
        ovxnode_id, 0, shape, dim_num, nullptr, 0, dtype);

    ovxtensor_map[node_id].push_back(tensor_id);
    soc_interface_SetGraphOutputTensor(tensor_id);
  }

  // Init virtual tensor
  for (const GraphTransferInfo::NodeOutputInfo& output_info :
       graph_transfer_info.node_output_info()) {
    const int node_id = output_info.node_id();
    // Skip input, output node.
    if ((std::find(graph_inputs.begin(), graph_inputs.end(), node_id)
            != graph_inputs.end()) ||
        (std::find(graph_outputs.begin(), graph_outputs.end(), node_id)
            != graph_outputs.end())) {
        continue;
    }
    const int count = output_info.max_byte_size_size();
    const uint32 ovxnode_id = ovxnode_map[node_id];
    //printf("Append %x\n",node_id);
    for (int i = 0; i < count; i++ ) {
      // TODO: Add shape and data type in output_info
      int dtype = 0;
      int auto_dim = 0;
      uint32 tensor_id = OVX_TENSOR_ID_NA;
      if (ovxnode_id != OVX_NODE_ID_NA) {
        tensor_id = soc_interface_AppendTensor(
          ovxnode_id, i, nullptr, auto_dim, nullptr, 0, dtype);
      }
      if (OVX_TENSOR_ID_NA == tensor_id) {
          ret = false;
      }
      ovxtensor_map[node_id].push_back(tensor_id);
    }
  }

  // Build input ports
  for (const GraphTransferInfo::NodeInputInfo& input_info :
       graph_transfer_info.node_input_info()) {
    const int node_id = input_info.node_id();
    const int count = input_info.node_input_size();
    auto node = ovxtensor_map[node_id];
    for (int i = 0; i < count; ++i) {
      const GraphTransferInfo::NodeInput& node_input =
          input_info.node_input(i);
      int input_id = node_input.node_id();
    if (input_ports_map.find(input_id) != input_ports_map.end()) {
        input_ports_map.emplace(input_id, std::vector<std::tuple<int, int>>());
      }
      input_ports_map[input_id].push_back(std::make_tuple(node_id, i));
      auto got = ovxtensor_map.find(input_id);
      if (got != ovxtensor_map.end()) {
        uint32 ovxnode_id = ovxnode_map[node_id];
        if (OVX_NODE_ID_NA == ovxnode_id) {
            continue;
        }
        uint32 ovxtensor_id = got->second[i];
        soc_interface_SetNodeInput(ovxnode_id, ovxtensor_id, i);
      }
    }
  }

  // Init const tensor
  for (const GraphTransferInfo::ConstNodeInfo& tensor :
       graph_transfer_info.const_node_info()) {
    const int node_id = tensor.node_id();
    //TODO: Check attr
    CHECK(tensor.shape_size() <= OVX_MAX_DIM_NUM);
    uint32 shape[OVX_MAX_DIM_NUM] = { 0 };

    for(int i = 0; i < tensor.shape_size(); i++) {
        shape[i] = tensor.shape(i);
    }
    int dtype = 0;
    auto ports = input_ports_map[node_id];
    for (auto pair : ports) {
      uint32 ovxnode_id = ovxnode_map[std::get<0>(pair)];
      uint32 p = std::get<1>(pair);
      uint32 tensor_id = soc_interface_AppendConstTensor(
          tensor.name().c_str(),
          ovxnode_id, p,
          shape, tensor.shape_size(),
          (uint8*)tensor.data().data(), tensor.data().length(),
          dtype);
      if (tensor_id != OVX_TENSOR_ID_NA) {
        ovxtensor_map[node_id].push_back(tensor_id);
      }
    }
  }

  LOG(INFO) << "Setup graph completed";

  if (false == ret) {
    LOG(INFO) << "Setup graph fail";
    soc_interface_TeardownGraph();
  } else {
    // construct graph
    ret = soc_interface_ConstructGraph();
  }
  return ret;

}

bool OvxControlWrapper::ExecuteGraph() {
  bool ret = soc_interface_ExecuteGraph();
  //printf("Exec finish\n");
  return ret;
}

bool OvxControlWrapper::TeardownGraph() {
  return soc_interface_TeardownGraph();
}

bool OvxControlWrapper::ReadOutputNode(
    const string& node_name, TensorAllocatorFunc tensor_allocator) {
  CHECK_NE(execute_info_, nullptr);
  //printf("Read Out\n");
  TensorShape output_shape;
  for (int i = 0; i < execute_info_->graph_output_node_name_size(); ++i) {
    if (execute_info_->graph_output_node_name(i) == node_name) {
      for (const TensorShapeProto::Dim& dim :
           execute_info_->default_graph_output_tensor_shape(i).shape().dim()) {
        output_shape.AddDim(dim.size());
      }
      break;
    }
  }
  std::vector<IRemoteFusedGraphExecutor::ByteArray> outputs;
  ReadOutputNode(node_name, &outputs);
  Tensor* output = tensor_allocator(output_shape);
  CHECK(output->TotalBytes() >= std::get<1>(outputs[0]));
  // TODO: Avoid specifying float
  std::memcpy(output->flat<float>().data(), std::get<0>(outputs[0]),
              std::get<1>(outputs[0]));
  return true;
}

bool OvxControlWrapper::ReadOutputNode(
    const string& node_name, std::vector<ByteArray>* const outputs) {
  CHECK(outputs != nullptr);
  ByteArray output;
  soc_interface_ReadOutputNode(node_name.c_str(), &std::get<0>(output),
                                    (uint64_t*)&std::get<1>(output));
  //std::get<2>(output) = DT_FLOAT;
  std::get<2>(output) = DT_BFLOAT16;
  outputs->emplace_back(output);
  return true;
}

bool OvxControlWrapper::FillInputNode(const string& node_name,
                                          const Tensor& tensor) {
  StringPiece tensor_data = tensor.tensor_data();
  LOG(INFO) << "Input tensor data: element size = " << tensor.NumElements()
            << ", byte syze = " << tensor.TotalBytes();
  uint32 tensor_id = 0;
  soc_interface_FillInputTensor(tensor_id, (uint8*)tensor_data.data(), tensor_data.size());
  return true;
  //return true;
}

#else
int OvxControlWrapper::GetVersion() { return -1; }
bool OvxControlWrapper::Init(const RemoteFusedGraphExecuteInfo&) {
  return false;
}
bool OvxControlWrapper::Finalize() { return false; }
bool OvxControlWrapper::SetupGraph() { return false; }
bool OvxControlWrapper::ExecuteGraph() { return false; }
bool OvxControlWrapper::TeardownGraph() { return false; }
bool OvxControlWrapper::FillInputNode(const string&, const Tensor&) {
  return false;
}
bool OvxControlWrapper::ReadOutputNode(
    const string& node_name, TensorAllocatorFunc tensor_allocator) {
  return false;
}
bool OvxControlWrapper::ReadOutputNode(const string&,
                                           std::vector<ByteArray>* const) {
  return false;
}
#endif

}  // namespace tensorflow
