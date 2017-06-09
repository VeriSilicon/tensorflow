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

#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "ovx_controller.h"
#include "ovx_log.h"

#define _clean_hash(d)               (d = 0)
#define _hash_set_op(op)             (((uint64_t)op & 0xFFFFFFFF) << 32)
#define _hash_set_port(port)         ((uint64_t)port & 0xFFFFFFFF)
#define _hash_op_port(op, port)      (_hash_set_op(op) | _hash_set_port(port))

#define ATTR_CONV_STRIDE                _hash_op_port(VSI_NN_OP_CONV2D, 0x3)
#define ATTR_CONV_PAD                   _hash_op_port(VSI_NN_OP_CONV2D, 0x4)

#define ATTR_POOL_STRIDE                _hash_op_port(VSI_NN_OP_POOL, 0x1)
#define ATTR_POOL_KSIZE                 _hash_op_port(VSI_NN_OP_POOL, 0x2)
#define ATTR_POOL_PAD                   _hash_op_port(VSI_NN_OP_POOL, 0x3)
#define ATTR_POOL_TYPE                  _hash_op_port(VSI_NN_OP_POOL, 0x4)

#define ATTR_CONV_RELU_STRIDE           _hash_op_port(VSI_NN_OP_CONV_RELU, 0x3)
#define ATTR_CONV_RELU_PAD              _hash_op_port(VSI_NN_OP_CONV_RELU, 0x4)

#define ATTR_CONV_RELU_POOL_STRIDE_POOL _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x3)
#define ATTR_CONV_RELU_POOL_KSIZE_POOL  _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x4)
#define ATTR_CONV_RELU_POOL_PAD_POOL    _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x5)
#define ATTR_CONV_RELU_POOL_POOL_TYPE   _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x6)
#define ATTR_CONV_RELU_POOL_STRIDE_CONV _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x7)
#define ATTR_CONV_RELU_POOL_PAD_CONV    _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x8)

#define ATTR_CONCAT_DIM                 _hash_op_port(VSI_NN_OP_CONCAT, 0x0)

#define ATTR_LRN_ALPHA                  _hash_op_port(VSI_NN_OP_LRN, 0x1)
#define ATTR_LRN_BETA                   _hash_op_port(VSI_NN_OP_LRN, 0x2)
#define ATTR_LRN_BIAS                   _hash_op_port(VSI_NN_OP_LRN, 0x3)
#define ATTR_LRN_DEPTH_RADIUS           _hash_op_port(VSI_NN_OP_LRN, 0x4)

// Const tensor
#define TENSOR_CONV_WEIGHTS             _hash_op_port(VSI_NN_OP_CONV2D, 0x1)
#define TENSOR_CONV_RELU_WEIGHTS        _hash_op_port(VSI_NN_OP_CONV_RELU, 0x1)
#define TENSOR_CONV_RELU_POOL_WEIGHTS   _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x1)
#define TENSOR_CONV_BIAS                _hash_op_port(VSI_NN_OP_CONV2D, 0x2)
#define TENSOR_CONV_RELU_BIAS           _hash_op_port(VSI_NN_OP_CONV_RELU, 0x2)
#define TENSOR_CONV_RELU_POOL_BIAS      _hash_op_port(VSI_NN_OP_CONV_RELU_POOL, 0x2)

#define TENSOR_FCL_WEIGHTS              _hash_op_port(VSI_NN_OP_FCL, 0x1)
#define TENSOR_FCL_RELU_WEIGHTS         _hash_op_port(VSI_NN_OP_FCL_RELU, 0x1)
#define TENSOR_FCL_BIAS                 _hash_op_port(VSI_NN_OP_FCL, 0x2)
#define TENSOR_FCL_RELU_BIAS            _hash_op_port(VSI_NN_OP_FCL_RELU, 0x2)

const static uint32_t s_tf_dim[] = {0, 3, 1, 2};

static int process_tensor_port(vsi_nn_node_t* node,
        int port, uint32_t * shape, uint32_t dim_num,
        const uint8_t* const attr, int len) {
  int new_port = -1;
  switch (node->op) {
    case VSI_NN_OP_CONCAT:
      if (port > 0) {
        new_port = port - 1;
      }
      break;
    default:
      break;
  }
  return new_port;
}


int ovx_controller_read_attr(vsi_nn_node_t* node,
        int port, uint32_t * shape, uint32_t dim_num,
        const uint8_t* const attr, int len) {
  int new_port = -1;
  uint64_t hash;
  _clean_hash(hash);
  hash =_hash_op_port(node->op, port);
  switch (hash) {
    // Attributes
    case ATTR_CONV_STRIDE:
    case ATTR_CONV_RELU_STRIDE:
    case ATTR_CONV_RELU_POOL_STRIDE_CONV:
      node->nn_param.conv2d.stride[0] = shape[1];
      node->nn_param.conv2d.stride[1] = shape[2];
      break;
    case ATTR_POOL_STRIDE:
    case ATTR_CONV_RELU_POOL_STRIDE_POOL:
      node->nn_param.pool.stride[0] = shape[1];
      node->nn_param.pool.stride[1] = shape[2];
      break;
    case ATTR_POOL_KSIZE:
    case ATTR_CONV_RELU_POOL_KSIZE_POOL:
      node->nn_param.pool.ksize[0] = shape[1];
      node->nn_param.pool.ksize[1] = shape[2];
      break;
    case ATTR_POOL_PAD:
    case ATTR_CONV_RELU_POOL_PAD_POOL:
      node->nn_param.pool.pad_type = (uint32_t)((int32_t*)attr)[0];
      break;
    case ATTR_CONV_PAD:
    case ATTR_CONV_RELU_PAD:
    case ATTR_CONV_RELU_POOL_PAD_CONV:
      node->nn_param.conv2d.pad_type = (uint32_t)((int32_t*)attr)[0];
      break;
    case ATTR_CONCAT_DIM:
      node->nn_param.concat.axis = s_tf_dim[((uint32_t *)attr)[0]];
      break;
    case ATTR_LRN_ALPHA:
      node->nn_param.lrn.alpha = ((float*)attr)[0];
      break;
    case ATTR_LRN_BETA:
      node->nn_param.lrn.beta = ((float*)attr)[0];
      break;
    case ATTR_LRN_BIAS:
      node->nn_param.lrn.bias = ((float*)attr)[0];
      break;
    case ATTR_LRN_DEPTH_RADIUS:
      node->nn_param.lrn.size = (uint32_t)((int32_t*)attr)[0];
      break;


    // Const tensor
    case TENSOR_CONV_WEIGHTS:
    case TENSOR_CONV_RELU_WEIGHTS:
    case TENSOR_CONV_RELU_POOL_WEIGHTS:
    case TENSOR_FCL_WEIGHTS:
    case TENSOR_FCL_RELU_WEIGHTS:
      new_port = 1;
      break;
    case TENSOR_CONV_BIAS:
    case TENSOR_CONV_RELU_BIAS:
    case TENSOR_CONV_RELU_POOL_BIAS:
    case TENSOR_FCL_BIAS:
    case TENSOR_FCL_RELU_BIAS:
      new_port = 2;
      break;
    default:
      // Is not attr
      OVXLOGE("Unknown port(%d) of op(%u)", port, node->op);
      break;
  }
  if (0 > new_port) {
    new_port = process_tensor_port( node, port, shape, dim_num, attr, len );
  }
  return new_port;
}

