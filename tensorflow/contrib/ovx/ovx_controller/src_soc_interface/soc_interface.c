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

#include <inttypes.h>

#include "soc_interface.h"
#include "ovx_controller.h"
#include "ovx_log.h"

int soc_interface_GetWrapperVersion() {
  OVXLOGD("GetWrapperVersion");
  return ovx_controller_GetWrapperVersion();
}

int soc_interface_GetSocControllerVersion() {
  OVXLOGD("GetSocControllerVersion");
  return ovx_controller_GetOvxBinaryVersion();
}

bool soc_interface_Init() {
  OVXLOGD("Init");
  return ovx_controller_InitOvx();
}

bool soc_interface_Finalize() {
  OVXLOGD("Finalize");
  return ovx_controller_DeInitOvx();
}

bool soc_interface_ExecuteGraph() {
  bool success;
  success = ovx_controller_ExecuteGraph();
  return success;
}

bool soc_interface_TeardownGraph() {
  OVXLOGD("TeardownGraph");
  return true;
}

bool soc_interface_FillInputTensor(uint32_t tensor_id,
        const uint8_t* const buf, uint64_t buf_size) {
  OVXLOGD("FillInputTensor %u, sz: %lu", tensor_id, buf_size);
  bool ret = ovx_controller_FillInputTensor(
          tensor_id, buf, buf_size);
  return ret;
}

uint64_t soc_interface_ReadOutputNode(
    const char* const node_name, uint8_t** buf, uint64_t *bytes) {
  OVXLOGD("ReadOutputNode %s", node_name);
  return ovx_controller_GetOutputNodeData(node_name, buf, bytes);
}

// Append const node to the graph
uint32_t soc_interface_AppendConstTensor(
    const char* const name,
    uint32_t node_id, int input_index,
    uint32_t * shape, uint32_t dim_num,
    const uint8_t* const data, int data_length,
    int dtype) {
  uint32_t id = (uint32_t)-1;
  OVXLOGI("Creat tensor const tensor %s", name);
  id = ovx_controller_AppendConstTensor(
                        name, node_id, input_index,
                        shape, dim_num, data, data_length,
                        dtype);
  return id;
}

// Append node to the graph
uint32_t soc_interface_AppendNode(
    const char* const name, int op_id) {
  const uint32_t ovxnode_id = ovx_controller_AppendNode(name, op_id);
  return ovxnode_id;
}

void soc_interface_SetNodeInput(
        uint32_t node_id, uint32_t tensor_id, int port) {
  ovx_controller_SetNodeInput(node_id, tensor_id, port);
}

uint32_t soc_interface_AppendTensor(
        uint32_t node_id, int port, uint32_t * shape, uint32_t dim_num,
        const uint8_t* const data, int data_length,
        int dtype) {
  OVXLOGI("Append tensor for node(%x, %d)", node_id, port);
  const uint32_t tensor_id = ovx_controller_AppendTensor(
          node_id, port, shape, dim_num, data, data_length, dtype);
  return tensor_id;
}

void soc_interface_SetGraphOutputTensor(uint32_t tensor_id) {
  ovx_controller_SetGraphOutputTensor(tensor_id, 0);
}

void soc_interface_SetGraphInputTensor(uint32_t tensor_id) {
  ovx_controller_SetGraphInputTensor(tensor_id, 0);
}

// Instantiate graph
bool soc_interface_InstantiateGraph(
        const int input_num, const int output_num,
        const int tensor_num, const int node_num) {
  const uint32_t nn_id = ovx_controller_InstantiateGraph(
          input_num, output_num,
          tensor_num, node_num);
  ovx_controller_SetTargetGraphId(nn_id);
  return true;
}

// Construct graph
bool soc_interface_ConstructGraph() {
  return ovx_controller_ConstructGraph();
}

void soc_interface_SetLogLevel(int log_level) {
  SetLogLevel(log_level);
}

void soc_interface_SetDebugFlag(uint64_t flag) {
  OVXLOGI("Set debug flag 0x%" PRIx64, flag);
}

