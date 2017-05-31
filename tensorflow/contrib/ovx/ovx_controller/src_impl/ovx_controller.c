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

#include <stdio.h>
#include <string.h>

#include "vsi_nn_graph.h"
#include "vsi_nn_math.h"
#include "vsi_nn_node.h"
#include "vsi_nn_util.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_node_attr_template.h"
#include "ovx_controller.h"
#include "ovx_log.h"


// if true, show id for each node
//static const bool DBG_SHOW_ID = false;

//static const uint32_t OUTPUT_PARAM_MAX_LINE_SIZE = 1000;

#define OVX_CONTROLLER_VERSION 0

// allocate print bufsize in advance @MB
//#define PRINT_BUFSIZE (2 * 1024 * 1024)

//static unsigned char s_print_buf[PRINT_BUFSIZE];

static vsi_nn_context_t s_context = NULL;
static vsi_nn_graph_t * s_graph = NULL;
static uint8_t *        s_output = NULL;

static void _data_fp32_to_fp16(
        float * src, int16_t * dst, int count) {
  for (int i = 0; i < count; i ++) {
    dst[i] = vsi_nn_Fp32toFp16(src[i]);
  }
}

static void _data_fp16_to_fp32(
        int16_t * src, float * dst, int count) {
  for (int i = 0; i < count; i ++) {
    dst[i] = vsi_nn_Fp16toFp32(src[i]);
  }
}

bool ovx_controller_ExecuteGraph() {
  vx_status status;
  OVXLOGI("Execute graph.");
  if (NULL == s_graph) {
      return false;
  }

  status = vsi_nn_RunGraph(s_graph);

  if (VX_SUCCESS != status) {
    OVXLOGE("Execution failed");
  }

  return (VX_SUCCESS == status);
}

uint32_t ovx_controller_GetTargetGraphId() {
  return 1;
}

void ovx_controller_SetTargetGraphId(uint32_t graph_id) {
    //TODO
}

void ovx_controller_PrintGraph(uint32_t id) {
  if (NULL == s_graph) {
    return;
  }

  vsi_nn_PrintGraph(s_graph);
}

int ovx_controller_GetWrapperVersion() {
  return OVX_CONTROLLER_VERSION;
}

int ovx_controller_GetOvxBinaryVersion() {
  int retval = 0;
  return retval;
}

bool ovx_controller_InitOvx() {
  s_output = NULL;
  s_context = vsi_nn_CreateContext();
  return (NULL != s_context);
}

bool ovx_controller_DeInitOvx() {
  OVXLOGI("Finalize ovx");
  if (NULL != s_output) {
    free(s_output);
    s_output = NULL;
  }
  if (NULL != s_graph) {
    vsi_nn_ReleaseGraph(&s_graph);
  }
  if (NULL != s_context) {
    vsi_nn_ReleaseContext(&s_context);
  }
  return true;
}

// Append const tensor to the graph
uint32_t ovx_controller_AppendConstTensor(
        const char* const name, uint32_t node_id, int port,
        uint32_t * shape, uint32_t dim_num,
        const uint8_t* const data, uint64_t data_length, int data_type) {
  vsi_nn_tensor_id_t tensor_id = VSI_NN_TENSOR_ID_NA;
  vsi_nn_node_t * node = NULL;
  int new_port;
  node = vsi_nn_GetNode(s_graph, node_id);
  if (NULL == node) {
      return tensor_id;
  }
  new_port = ovx_controller_read_attr(node, port,
          shape, dim_num, data, data_length);
  if (new_port > 0) {
    tensor_id = ovx_controller_AppendTensor(
            VSI_NN_NODE_ID_NA, 0,
            shape, dim_num,
            data, data_length, data_type);
    if (VSI_NN_TENSOR_ID_NA == tensor_id) {
      OVXLOGE("Create const tensor %s fail.", name);
    }
    //printf("new %d [%d]--> %d \n",node_id, new_port, tensor_id);
    node->input.tensors[new_port] = tensor_id;
  }
  return tensor_id;
}


// Append node to the graph
uint32_t ovx_controller_AppendNode(
    const char* const name, int op_id) {
  vsi_nn_node_id_t ovxnode_id = 0;
  vsi_nn_node_t * node = NULL;

  OVXLOGI("Append node %s(%d).", name, op_id);

  node = vsi_nn_AppendNode(s_graph, op_id, &ovxnode_id);
  if (NULL == node) {
    OVXLOGE("Failed to append node %s(%d)", name, op_id);
    return ovxnode_id;
  }
  vsi_nn_apply_node_attr_template(node);

  return ovxnode_id;
}

bool ovx_controller_ConstructGraph() {
  vx_status status;
  if (NULL == s_graph) {
    return false;
  }

  vsi_nn_PrintGraph(s_graph);
  status = vsi_nn_SetupGraph(s_graph, true);
  if (VX_SUCCESS == status) {
    status = vsi_nn_VerifyGraph(s_graph);
  } else {
    OVXLOGE("Setup graph fail.");
  }
  return (VX_SUCCESS == status);
}

uint32_t ovx_controller_InstantiateGraph(
        uint32_t input_num, uint32_t output_num,
        uint32_t tensor_num, uint32_t node_num) {
  OVXLOGI("Creat graph(%u, %u) tensor(%u) node(%u)",
          input_num, output_num,
          tensor_num, node_num);
  s_graph = vsi_nn_CreateGraph(s_context, tensor_num, node_num);
  if (NULL == s_graph) {
    OVXLOGE("Create graph(%d, %d) fail.", tensor_num, node_num);
  } else {
      vsi_nn_SetGraphInputs(s_graph, NULL, input_num);
      vsi_nn_SetGraphOutputs(s_graph, NULL, output_num);
  }
  return 1;
}

uint64_t ovx_controller_GetOutputNodeData(const char* node_name,
        uint8_t** buf, uint64_t* bytes) {

  int16_t* fp16_out;
  uint64_t sz;
  OVXLOGI("Read output of %s.", node_name);
  //TODO: Find graph by name.
  //TODO: Find tensor by name.
  //*bytes = vsi_nn_CopyTensorToBuffer(s_graph, s_graph->tensors[1], *buf);
  fp16_out = (int16_t*)vsi_nn_ConvertTensorToData(s_graph, s_graph->tensors[1]);
  if (NULL != s_output) {
      free(s_output);
      s_output = NULL;
  }
  if (NULL != fp16_out) {
    sz = 10 * sizeof(float);
    s_output = (uint8_t*)malloc(sz);
    _data_fp16_to_fp32(fp16_out, (float*)s_output, sz / sizeof(float));
    free(fp16_out);
    *buf = s_output;
    *bytes = sz;
    //printf("%f, %f, %f, %f\n",
    //        ((float*)s_output)[0],
    //        ((float*)s_output)[1],
    //        ((float*)s_output)[5],
    //        ((float*)s_output)[9]);
  }
  return *bytes;
}

bool ovx_controller_FillInputTensor(uint32_t tensor_id,
        const uint8_t* const buf, uint64_t buf_size) {
  vx_status status = VX_FAILURE;
  vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(s_graph, tensor_id);
  if (NULL != tensor) {
    int16_t * fp16buf;
    printf("Convert to fp16 ...\n ");
    fp16buf = (int16_t*)malloc((buf_size / 4) * sizeof(uint16_t));
    _data_fp32_to_fp16((float*)buf, fp16buf, (buf_size / 4));
    status = vsi_nn_CopyDataToTensor(s_graph, tensor, (uint8_t*)fp16buf);
    free(fp16buf);
  }
  if (VX_SUCCESS != status) {
    OVXLOGE("Copy data(%lu) to tensor %u fail.", buf_size, tensor_id);
  }
  return (VX_SUCCESS == status);
}

void ovx_controller_SetNodeInput(
        uint32_t node_id, uint32_t tensor_id, int port) {
  vsi_nn_node_t * node = vsi_nn_GetNode(s_graph, node_id);
  if (NULL == node) {
    OVXLOGE("Find node %x fail.", node_id);
    return;
  }
  if (port < node->input.num) {
    node->input.tensors[port] = tensor_id;
  } else {
    OVXLOGE("Unsopport input port %d of node %x.", port, node_id);
  }
}

void  ovx_controller_SetGraphOutputTensor(uint32_t tensor_id, int port) {
  if (NULL == s_graph) {
      return;
  }
  if (port >= s_graph->output.num) {
    OVXLOGE("Unsopport output port %d, the max is %d",
            port, s_graph->output.num );
    return;
  }
  s_graph->output.tensors[port] = tensor_id;
}

void  ovx_controller_SetGraphInputTensor(uint32_t tensor_id, int port) {
  if (NULL == s_graph) {
      return;
  }
  if (port >= s_graph->input.num) {
    OVXLOGE("Unsopport input port %d, the max is %d",
            port, s_graph->input.num );
    return;
  }
  s_graph->input.tensors[port] = tensor_id;
}

uint32_t ovx_controller_AppendTensor(
        uint32_t node_id, int port,
        uint32_t * shape, uint32_t dim_num,
        const uint8_t* const data, int data_length,
        int dtype) {
  vsi_nn_tensor_id_t tensor_id;
  vsi_nn_tensor_attr_t attr;
  int16_t* fp16buf = NULL;
  uint8_t * input_buf = (uint8_t*)data;
  memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
  attr.dtype.vx_type = VX_TYPE_FLOAT16;
  if (NULL == shape || 0 == dim_num) {
    OVXLOGI("Create virtual tensor for node(%u).", node_id);
    attr.vtl = vx_true_e;
    attr.is_const = vx_false_e;
    attr.dim_num = VSI_NN_DIM_AUTO;
  } else {
    attr.vtl = vx_false_e;
    attr.dim_num = dim_num;
    memcpy(attr.size, shape, dim_num * sizeof(uint32_t));
    // TODO: Fix me
    if (NULL != input_buf && data_length > 0) {
      // Const tensor
      // For bias
      if (1 == attr.size[0] && 1 == attr.size[2]) {
        vsi_nn_SqueezeShape(attr.size, &attr.dim_num);
        attr.dtype.vx_type = VX_TYPE_FLOAT32;
      } else {
        // TODO: workaround
        printf("Convert to fp16 ...\n ");
        fp16buf = (int16_t*)malloc((data_length / 4) * sizeof(uint16_t));
        _data_fp32_to_fp16((float*)input_buf, fp16buf, (data_length / 4));
        input_buf = (uint8_t*)fp16buf;
      }
    } else {
      // Input Output
      attr.dim_num --;
      memmove(&attr.size[0], &attr.size[1], attr.dim_num * sizeof(uint32_t));
    }
  //printf("%d/len(%d), %d, %d, %d, %d\n", dim_num, data_length, shape[0], shape[1], shape[2], shape[3]);
  //printf("==>%d, %d, %d, %d, %d\n", attr.dim_num, attr.size[0], attr.size[1], attr.size[2], attr.size[3]);
  }
  //TODO: Data type
  tensor_id = vsi_nn_AddTensor(s_graph, VSI_NN_TENSOR_ID_AUTO, &attr, input_buf);
  if (NULL != fp16buf) {
    free(fp16buf);
  }
  if (VSI_NN_TENSOR_ID_NA == tensor_id) {
    OVXLOGE("Create tensor fail.");
  } else if (VSI_NN_NODE_ID_NA != node_id) {
    vsi_nn_node_t * node = vsi_nn_GetNode(s_graph, node_id);
    if (!node) {
      OVXLOGE("Missing node %u.", node_id);
    } else {
      node->output.tensors[port] = tensor_id;
    }
  }
  return tensor_id;
}

