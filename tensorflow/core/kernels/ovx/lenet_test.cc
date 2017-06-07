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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include <fstream>

namespace tensorflow {
namespace graph_transforms {

// Declared here so we don't have to put it in a public header.
Status RewriteQuantizedStrippedModelForOvx(
    const GraphDef& input_graph_def, const TransformFuncContext& context,
    GraphDef* output_graph_def);

namespace {

  TEST(ovxRewriteTransformLenetTest, BasicRun) {
  Scope root = tensorflow::Scope::NewRootScope();
#if 0

  // Create a simple graph that calculates convolution,relu,pool.

  Output input =
      ops::Placeholder(root.WithOpName("input_image"), DT_FLOAT);

  //Conv1
  Tensor conv1_wieght_data(DT_FLOAT, TensorShape({5, 5, 1, 20}));
  test::FillIota<float>(&conv1_wieght_data, 0.0f);

  Output conv1_weights_op =
  ops::Const(root.WithOpName("conv1_weights_op"), Input::Initializer(conv1_wieght_data));

  Output conv1_op = 
    ops::Conv2D(root.WithOpName("conv1"), input, conv1_weights_op, {1, 1, 1, 1}, "VALID");

  Tensor conv1_bias_data(DT_FLOAT, TensorShape({1, 1, 1, 20}));
  test::FillIota<float>(&conv1_bias_data, 0.0f);

  Output conv1_bias_op =
  ops::Const(root.WithOpName("conv1_bias_op"), Input::Initializer(conv1_bias_data));

  Output conv1_bias_add_op = 
    ops::Add(root.WithOpName("conv1_bias_add_op"), conv1_op, conv1_bias_op);

  Output conv1_relu_op = ops::Relu(root.WithOpName("conv1_relu_op"), conv1_bias_add_op);

  Output conv1_relu_pool_op = ops::MaxPool(root.WithOpName("conv1_max_pool_op"), conv1_relu_op,
                                 {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID");

  //Conv2
  Tensor conv2_wieght_data(DT_FLOAT, TensorShape({5, 5, 20, 50}));
  test::FillIota<float>(&conv1_wieght_data, 0.0f);

  Output conv2_weights_op =
  ops::Const(root.WithOpName("conv2_weights_op"), Input::Initializer(conv2_wieght_data));

  Output conv2_op =
    ops::Conv2D(root.WithOpName("conv2"), conv1_relu_pool_op, conv2_weights_op, {1, 1, 1, 1}, "VALID");

  Tensor conv2_bias_data(DT_FLOAT, TensorShape({1, 1, 1, 50}));
  test::FillIota<float>(&conv2_bias_data, 0.0f);

  Output conv2_bias_op =
  ops::Const(root.WithOpName("conv2_bias_op"), Input::Initializer(conv2_bias_data));

  Output conv2_bias_add_op =
    ops::Add(root.WithOpName("conv2_bias_add_op"), conv2_op, conv2_bias_op);

  Output conv2_relu_op = ops::Relu(root.WithOpName("conv2_relu_op"), conv2_bias_add_op);

  Output conv2_max_pool_op = ops::MaxPool(root.WithOpName("conv2_max_pool_op"), conv2_relu_op,
                                 {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID");

  //FC1
  Output  fc1_input = ops::Reshape(root.WithOpName("fc1_reshape_op"),conv2_max_pool_op, {1, -1});
  Tensor fc1_wieght_data(DT_FLOAT, TensorShape({800, 500}));
  test::FillIota<float>(&fc1_wieght_data, 0.0f);

  Output fc1_weights_op =
  ops::Const(root.WithOpName("fc1_weights_op"), Input::Initializer(fc1_wieght_data));

  Output fc1_op = ops::MatMul(root.WithOpName("fc1_mul"), fc1_input, fc1_weights_op);

  Tensor fc1_bias_data(DT_FLOAT, TensorShape({1, 500}));
  test::FillIota<float>(&fc1_bias_data, 0.0f);

  Output fc1_bias_op =
  ops::Const(root.WithOpName("fc1_bias_op"), Input::Initializer(fc1_bias_data));

  Output fc1_bias_add_op =
    ops::Add(root.WithOpName("fc1_bias_add_op"), fc1_op, fc1_bias_op);

  Output fc1_relu_op = ops::Relu(root.WithOpName("fc1_relu_op"), fc1_bias_add_op);


  //FC2
  Output  fc2_input = ops::Reshape(root.WithOpName("fc2_reshape_op"),fc1_relu_op, {1, -1});

  Tensor fc2_wieght_data(DT_FLOAT, TensorShape({500, 10}));
  test::FillIota<float>(&fc2_wieght_data, 0.0f);

  Output fc2_weights_op =
  ops::Const(root.WithOpName("fc2_weights_op"), Input::Initializer(fc2_wieght_data));

  Output fc2_op = ops::MatMul(root.WithOpName("fc2_mul"), fc2_input, fc2_weights_op);

  Tensor fc2_bias_data(DT_FLOAT, TensorShape({1, 10}));
  test::FillIota<float>(&fc2_bias_data, 0.0f);

  Output fc2_bias_op =
  ops::Const(root.WithOpName("fc2_bias_op"), Input::Initializer(fc2_bias_data));

  Output fc2_bias_add_op =
    ops::Add(root.WithOpName("fc2_bias_add_op"), fc2_op, fc2_bias_op);

  Output fc2_relu_op = ops::Relu(root.WithOpName("fc2_relu_op"), fc2_bias_add_op);

  Output fc2_prob_op = ops::Softmax(root.WithOpName("fc2_prob_op"), fc2_relu_op);


  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

#else
  GraphDef graph_def;
  Status load_status = ReadBinaryProto(Env::Default(), "Lenet_2.pb", &graph_def);
  ASSERT_TRUE(load_status.ok());
//#else
//  GraphDef graph_def;
//  GraphDef fused_graph;
//  auto input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
//  auto data1 = ops::Const(root.WithOpName("d1"), {{3.f, 2.f},{-1.f, 0.f}});
//  //auto data2 = ops::Const(root.WithOpName("d2"), {{3.f, 1.f},{1.f, 2.f}});
//  auto add = ops::Add(root.WithOpName("add"), data1, input);
//
//  TransformFuncContext context;
//  context.input_names = {"input"};
//  context.output_names = {"add"};
//  context.params.insert(std::pair<string, std::vector<string>>(
//      {"input_shape0", {string("2, 2")}}));
//  context.params.insert(std::pair<string, std::vector<string>>(
//      {"input_type0", {string("float")}}));
//  TF_ASSERT_OK(root.ToGraphDef(&graph_def));
//  TF_ASSERT_OK(
//      RewriteQuantizedStrippedModelForOvx(graph_def, context, &fused_graph));
#endif

#if 1
  GraphDef fused_graph;
  TransformFuncContext context;
  context.input_names = {"input_image"};
  context.output_names = {"fc2_prob_op"};
  context.params.insert(std::pair<string, std::vector<string>>(
      {"input_shape0", {string("1,28,28,1")}}));
  context.params.insert(std::pair<string, std::vector<string>>(
      {"input_type0", {string("float")}}));
  TF_ASSERT_OK(
      RewriteQuantizedStrippedModelForOvx(graph_def, context, &fused_graph));
#endif

  WriteTextProto(Env::Default(), "/home/ubuser/Lenet_OrigionGraph.txt", graph_def);
  WriteTextProto(Env::Default(), "/home/ubuser/Lenet_RemoteFuseGraphHEXRewrite.txt", fused_graph);

  // 5.3 Setup session
  std::vector<Tensor> output_tensors;
  SessionOptions session_options;
  session_options.env = Env::Default();
  std::unique_ptr<Session> session =
      std::unique_ptr<Session>(NewSession(session_options));
  Status status = session->Create(fused_graph);
  ASSERT_TRUE(status.ok());
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;

  // 5.4 Setup input
  //Tensor input_a(DT_FLOAT, {1, 28, 28, 1});
  //test::FillIota<float>(&input_a, 0.0f);

  std::ifstream fin("2_1.raw", std::ios::binary);
  char data[28*28] = {-1};
  fin.read(data, 28*28);

  Tensor input_a(DT_FLOAT, TensorShape({1, 28, 28, 1}));
  for(int i = 0; i < 28*28; i++) {
    input_a.flat<float>().data()[i] = (float) ((unsigned char)data[i]) / 256.f;
#if 0
    printf("%1.3f, ",input_a.flat<float>().data()[i]);
    //printf("%3d, ",data[i]);
    if((i + 1 ) % 28 == 0)printf("\n");
#endif
  }

  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back("input_image", input_a);

  // 5.5 Setup output
  std::vector<string> outputs;
  outputs.emplace_back("remote_fused_graph_execute_node");

  // 5.6 Run inference with all node as output
  status = session->Run(run_options, inputs, outputs, {}, &output_tensors,
                        &run_metadata);


  Tensor out = output_tensors[0];
  printf("output: ");
  for (int i = 0; i < out.NumElements(); i ++) {
    printf("%f, ", ((float*)out.tensor_data().data())[i]);
  }
  printf("\n");
  ASSERT_TRUE(status.ok());

  // 5.7 Check output tensor value
  ASSERT_EQ(1, output_tensors.size());


}

}  // namespace
}  // namespace graph_transforms
}  // namespace tensorflow
