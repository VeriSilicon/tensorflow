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
#include "tensorflow/core/platform/png.h"
#include "tensorflow/core/lib/png/png_io.h"
#include <fstream>
#include <stdio.h>

namespace tensorflow {
namespace graph_transforms {

// Declared here so we don't have to put it in a public header.
Status RewriteQuantizedStrippedModelForOvx(
    const GraphDef& input_graph_def, const TransformFuncContext& context,
    GraphDef* output_graph_def);

namespace {

  TEST(ovxGoogleInceptionTest, BasicRun) {
  Scope root = tensorflow::Scope::NewRootScope();
  GraphDef graph_def;
  Status load_status = ReadBinaryProto(Env::Default(), "/mnt/shared/tensorflow_inception_graph.pb", &graph_def);
  LOG(WARNING)<<load_status.error_message();
  ASSERT_TRUE(load_status.ok());

#if 1
  GraphDef fused_graph;
  TransformFuncContext context;
  context.input_names = {"input"};
  context.output_names = {"output"};
  context.params.insert(std::pair<string, std::vector<string>>(
      {"input_shape0", {string("1,224,224,3")}}));
  context.params.insert(std::pair<string, std::vector<string>>(
      {"input_type0", {string("float")}}));
  TF_ASSERT_OK(
      RewriteQuantizedStrippedModelForOvx(graph_def, context, &fused_graph));
#endif

  WriteTextProto(Env::Default(), "./GoogleInception_OrigionGraph.txt", graph_def);
  WriteTextProto(Env::Default(), "./GoogleInception_RemoteFuseGraph.txt", fused_graph);

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

  //Read Png file.
//  FILE* fn = fopen("/mnt/shared/X_Pic.png", "rb");
//  fseek(fn, 0L, SEEK_END);
//  size_t fsize = ftell(fn);
//  fseek(fn, 0L, SEEK_SET);
//  printf("fsize %d\n", fsize);
//  char* data = new char[fsize];
//  printf("Read Size %d \n",fread(data, 1, fsize, fn));
//  fclose(fn);

//  png::DecodeContext decode;
//  int w = -1, h = -1, c = -1, cb = -1;
//  png::DecodeHeader(data, &w, &h, &c, &cb, nullptr);
//  printf("w %d, h %d, c %d, cb %d\n", w, h, c, cb);
//  png::CommonInitDecode(data, 3, 8, &decode);
//  char* image_buffer = new char[3*decode.width*decode.height];
//  png::CommonFinishDecode(reinterpret_cast<png_bytep>(image_buffer), 3*decode.width, &decode);
//  png::CommonFreeDecode(&decode);
  unsigned char* m_bgra = nullptr;
  do
  {
    //对PNG文件的解析
   // try to open file
   FILE* file = fopen("/mnt/shared/X_Pic.png", "rb");
   // unable to open
   if (file == 0) {
     printf("B0\n");
     break;
   }

   // create read struct
   png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
   // check pointer
   if (png_ptr == 0)
   {
    fclose(file);
    printf("B1\n");
    break;
   }
   // create info struct
   png_infop info_ptr = png_create_info_struct(png_ptr);
   // check pointer
   if (info_ptr == 0)
   {
    png_destroy_read_struct(&png_ptr, 0, 0);
    fclose(file);
    printf("B2\n");
    break;
   }
   // set error handling
   if (setjmp(png_jmpbuf(png_ptr)))
   {
    png_destroy_read_struct(&png_ptr, &info_ptr, 0);
    fclose(file);
    printf("B3\n");
    break;
   }
   // I/O initialization using standard C streams
   png_init_io(png_ptr, file);

   // read entire image , ignore alpha channel，如果你要使用alpha通道，请把PNG_TRANSFORM_STRIP_ALPHA去掉
   png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_EXPAND | PNG_TRANSFORM_STRIP_ALPHA, 0);
   /*
   PNG_TRANSFORM_EXPAND有下边几个处理：
   1.Expand paletted colors into true RGB triplets
   2.Expand grayscale images to full 8 bits from 1, 2, or 4 bits/pixel
   3.Expand paletted or RGB images with transparency to full alpha channels so the data will be available
     as RGBA quartets。
   PNG_TRANSFORM_STRIP_ALPHA：Strip alpha bytes from the input data without combining withthe background
   */
   int width =  info_ptr->width;
   int height = info_ptr->height;
   int color_type = info_ptr->color_type;
   int bit_depth = info_ptr->pixel_depth;
   png_bytep* row_pointers = png_get_rows(png_ptr,info_ptr);
   int pos=0;
   if(color_type == PNG_COLOR_TYPE_GRAY)
   {//对灰度图的处理
    unsigned char* m_8bit = new unsigned char[width*height];
    memset(m_8bit,0,width*height);
    for(int i=0;i<height;i++)
    {
     for(int j=0;j<width;j+=1)
     {
      m_8bit[pos++] = row_pointers[i][j];
     }
    }
   }
   else
   {//对非灰度图的处理
    m_bgra = new unsigned char[width*height*3];
    memset(m_bgra,0,width*height*3);
    for(int i=0;i<height;i++)
    {
     for(int j=0;j<3*width;j+=3)
     {
      m_bgra[pos++] = row_pointers[i][j+2];//BLUE
      m_bgra[pos++] = row_pointers[i][j+1];//GREEN
      m_bgra[pos++] = row_pointers[i][j];//RED
     }
    }
   }
   // free memory
   png_destroy_read_struct(&png_ptr, &info_ptr, 0);
   // close file
   fclose(file);
  }while(false);

  //Pre-Process
  //Processed Pixel Value= (Pixel Value - Mean) / Std
  //Mean = 117, Std = 1
  float* floatValues = new float[224 * 224 * 3];
  for (int i = 0; i < 224 * 224 * 3; i++) {
    //floatValues[i] = (image_buffer[i] - 117) / 1;
    floatValues[i] = (m_bgra[i] - 117) / 1;
  }



  Tensor input_a(DT_FLOAT, TensorShape({1, 224, 224, 3}));
  for(int i = 0; i < 224 * 224 * 3; i++) {
    input_a.flat<float>().data()[i] = floatValues[i];
#if 0
    printf("%1.3f, ",input_a.flat<float>().data()[i]);
    //printf("%3d, ",data[i]);
    if((i + 1 ) % 224 == 0)printf("\n");
#endif
  }

  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back("input", input_a);

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
