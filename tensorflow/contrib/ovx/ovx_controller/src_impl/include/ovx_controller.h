/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef GEMM_WRAPPER_H
#define GEMM_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "vsi_nn_node.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define OVX_CONTROLLER_GRAPH_ID_NA  ((uint32_t)-1)

// General functions
void ovx_controller_PrintGraph(uint32_t graph_id);

int ovx_controller_GetWrapperVersion();

int ovx_controller_GetOvxBinaryVersion();

bool ovx_controller_DeInitOvx();

bool ovx_controller_InitOvx();

uint32_t ovx_controller_GetTargetGraphId();

void ovx_controller_SetTargetGraphId(uint32_t graph_id);

bool ovx_controller_FillInputTensor(uint32_t tensor_id,
        const uint8_t* const buf, uint64_t buf_size);

uint64_t ovx_controller_GetOutputNodeData(const char* node_name,
        uint8_t** buf, uint64_t* bytes);

// Graph functions
uint32_t ovx_controller_InstantiateGraph(uint32_t input_num, uint32_t output_num,
        uint32_t tensor_num, uint32_t node_num);

void ovx_controller_InitGraph(int version);

bool ovx_controller_ConstructGraph();

bool ovx_controller_ExecuteGraph();

void ovx_controller_DumpNodeName(const uint32_t graph_id);

uint32_t ovx_controller_AppendNode(const char* const name, int op_id);

uint32_t ovx_controller_AppendConstTensor(const char* const name,
                                       uint32_t node_id, int input_index,
                                       uint32_t * shape, uint32_t dim_num,
                                       const uint8_t* const data, uint64_t data_length,
                                       int data_type);

void ovx_controller_SetNodeInput(uint32_t node_id, uint32_t tensor_id, int port);

void  ovx_controller_SetGraphOutputTensor(uint32_t tensor_id, int port);

void  ovx_controller_SetGraphInputTensor(uint32_t tensor_id, int port);

uint32_t ovx_controller_AppendTensor(uint32_t node_id, int port,
        uint32_t * shape, uint32_t dim_num,
        const uint8_t* const data, int data_length,
        int dtype);

int ovx_controller_read_attr(vsi_nn_node_t* node,
        int port, uint32_t * shape, uint32_t dim_num,
        const uint8_t* const attr, int len);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // GEMM_WRAPPER_H

