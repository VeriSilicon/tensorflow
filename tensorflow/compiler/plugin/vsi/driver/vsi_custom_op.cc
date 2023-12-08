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

#include "tensorflow/compiler/plugin/vsi/driver/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/vsi/driver/xla_vsi_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace tensorflow {

class LogSoftmaxOp : public XlaOpKernel {
 public:
  explicit LogSoftmaxOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto input = ctx->Input(0);
    TensorShape tensor_input_shape = ctx->InputShape(0);
    xla::PrimitiveType data_type = ctx->input_xla_type(0);
    xla::Shape output_shape =
        TensorShapeToXLAShape(data_type, tensor_input_shape);
    xla::XlaOp output;
    std::vector<xla::XlaOp> args = {input};
    xla::XlaBuilder& b = *ctx->builder();

    // the axis parameter actually not been used, it's always zero.
    output = xla::CustomCall(&b, "CustomCallLogSoftmax", args,
                             output_shape, "");

    ctx->SetOutput(0, output);
  }
};
REGISTER_XLA_OP(Name("LogSoftmax").Device(DEVICE_VSI_NPU_XLA_JIT), LogSoftmaxOp);

class TopKOp : public XlaOpKernel {
 public:
  explicit TopKOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted_));
  }

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    int last_dim = input_shape.dims() - 1;
    int last_dim_size = input_shape.dim_size(last_dim);

    int64_t k;
    bool k_bound_inferrable =
        context
            ->ConstantInputAsIntScalar(1, &k,
                                       xla::ValueInferenceMode::kUpperBound)
            .ok();
    if (!k_bound_inferrable) {
      // - If we can infer the bound of K, use the bound.
      // - If not, use last dim's size.
      k = last_dim_size;
    }
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));

    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_shape.DebugString()));

    OP_REQUIRES(
        context, last_dim_size >= k,
        errors::InvalidArgument("input must have at least k columns. Had ",
                                last_dim_size, ", needed ", k));
    if (last_dim_size < k) {
      k = last_dim_size;
    }

    bool k_is_dynamic;
    OP_REQUIRES_OK(context,
                   context->ResolveInputDynamismIntoPred(1, &k_is_dynamic));
    
    std::vector<xla::XlaOp> args = {context->Input(0)};
    xla::XlaBuilder& b = *context->builder();

    TensorShape in_shape = input_shape;
    in_shape.set_dim(last_dim, k);
    xla::Shape value_shape = TensorShapeToXLAShape(
        context->input_xla_type(0), in_shape);

    xla::Shape indices_shape = TensorShapeToXLAShape(
        context->input_xla_type(1), in_shape);

    xla::Shape output_shape = xla::ShapeUtil::MakeTupleShape({value_shape, indices_shape});

    attribute_map_.AddAttribute("k", k);
    attribute_map_.AddAttribute("sorted", sorted_);

    xla::XlaOp output_tuple = xla::CustomCall(&b, "CustomCallTopK", args,
        output_shape, attribute_map_.Serialise());

    auto values = xla::GetTupleElement(output_tuple, 0);
    auto indices = xla::GetTupleElement(output_tuple, 1);
    if (k_is_dynamic) {
      xla::XlaOp dynamic_k = context->Input(1);
      values = xla::SetDimensionSize(values, dynamic_k, last_dim);
      indices = xla::SetDimensionSize(indices, dynamic_k, last_dim);
    }
    context->SetOutput(0, values);
    context->SetOutput(1, indices);
  }

 private:
  bool sorted_;
  xla::vsiplugin::AttributeMap attribute_map_;
};

REGISTER_XLA_OP(Name("TopKV2").CompileTimeConstantInput("k").TypeConstraint(
                    "T", {DT_UINT32, DT_INT32, DT_UINT64, DT_INT64, DT_FLOAT,
                          DT_HALF, DT_DOUBLE, DT_BFLOAT16})
                          .Device(DEVICE_VSI_NPU_XLA_JIT),
                TopKOp);

class SqueezeOp : public XlaOpKernel {
 public:
  explicit SqueezeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    std::vector<int32> squeeze_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("squeeze_dims", &squeeze_dims));
    squeeze_dims_.insert(squeeze_dims.begin(), squeeze_dims.end());
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    auto existing_dims = input_shape.dim_sizes();
    int existing_dims_size = input_shape.dims();
    std::vector<int64_t> new_shape;

    std::unordered_set<int32> wrapped_squeeze_dims;
    wrapped_squeeze_dims.reserve(squeeze_dims_.size());
    // Validate squeeze dims against the input.
    for (int32_t dim : squeeze_dims_) {
      OP_REQUIRES(ctx, (dim >= -input_shape.dims() && dim < input_shape.dims()),
                  errors::InvalidArgument("Tried to squeeze dim index ", dim,
                                          " for tensor with ",
                                          input_shape.dims(), " dimensions."));
      // If dim is < 0, we wrap around (-1 means the last element).
      if (dim < 0) {
        dim = existing_dims_size + dim;
      }

      wrapped_squeeze_dims.insert(dim);
    }

    std::vector<int64_t> squeeze_dims_all(wrapped_squeeze_dims.begin(),
                                          wrapped_squeeze_dims.end());
    for (int i = 0; i < existing_dims_size; ++i) {
      auto existing_dim = existing_dims[i];

      // If squeeze_set is non-empty, only squeeze those dimensions.
      if (!wrapped_squeeze_dims.empty()) {
        if (wrapped_squeeze_dims.count(i) > 0) {
          OP_REQUIRES(ctx, existing_dim == 1,
                      errors::InvalidArgument(
                          "Tried to explicitly squeeze dimension ", i,
                          " but dimension was not 1: ", existing_dim));
        } else {
          // This dimension is not being squeezed.
          new_shape.push_back(existing_dim);
        }
      } else {
        // Copy over all non-1-length dimensions.
        LOG(INFO) << "CCQQ existing_dim = " << existing_dim;
        if (existing_dim != 1) {
          new_shape.push_back(existing_dim);
        } else {
          squeeze_dims_all.push_back(i);
        }
      }
    }

    attribute_map_.AddAttribute("squeeze_dims", squeeze_dims_all);
    xla::Shape output_shape = xla::ShapeUtil::MakeShape(
        ctx->input_xla_type(0), new_shape);
    auto input = ctx->Input(0);
    std::vector<xla::XlaOp> args = {input};
    xla::XlaBuilder& b = *ctx->builder();
    xla::XlaOp output = xla::CustomCall(&b, "CustomCallSqueeze", args,
                             output_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  std::unordered_set<int32> squeeze_dims_;
  xla::vsiplugin::AttributeMap attribute_map_;
};

REGISTER_XLA_OP(Name("Squeeze").Device(DEVICE_VSI_NPU_XLA_JIT), SqueezeOp);

class SoftmaxOp : public XlaOpKernel {
 public:
  explicit SoftmaxOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto input = ctx->Input(0);
    TensorShape tensor_input_shape = ctx->InputShape(0);
    xla::PrimitiveType data_type = ctx->input_xla_type(0);
    xla::Shape output_shape =
        TensorShapeToXLAShape(data_type, tensor_input_shape);
    xla::XlaOp output;
    std::vector<xla::XlaOp> args = {input};
    xla::XlaBuilder& b = *ctx->builder();

    // the axis parameter actually not been used, it's always zero.
    output = xla::CustomCall(&b, "CustomCallSoftmax", args,
                             output_shape, "");

    ctx->SetOutput(0, output);
  }
};
// REGISTER_XLA_OP(Name("Softmax").Device(DEVICE_VSI_NPU_XLA_JIT), SoftmaxOp);

class ArgMaxOp : public XlaOpKernel {
 public:
  explicit ArgMaxOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto input = ctx->Input(0);
    TensorShape tensor_input_shape = ctx->InputShape(0);
    const TensorShape dimension_shape = ctx->InputShape(1);
    xla::PrimitiveType data_type = ctx->input_xla_type(0);
    xla::Shape output_shape =
        TensorShapeToXLAShape(data_type, tensor_input_shape);
    xla::XlaOp output;
    std::vector<xla::XlaOp> args = {input};
    xla::XlaBuilder& b = *ctx->builder();

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dimension_shape),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension_shape.DebugString()));

    int64_t dim;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &dim));

    const int input_dims = tensor_input_shape.dims();
    const int axis = dim < 0 ? dim + input_dims : dim;

    OP_REQUIRES(
        ctx, axis >= 0 && axis < input_dims,
        errors::InvalidArgument("Expected dimension in the range [", -input_dims,
                                ", ", input_dims, "), but got ", dim));
    const int64_t axis_size = tensor_input_shape.dim_size(axis);
    OP_REQUIRES(
        ctx, axis_size > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                tensor_input_shape.DebugString()));

    attribute_map_.AddAttribute("dimension", axis);

    output = xla::CustomCall(&b, "CustomCallArgMax", args,
                             output_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  xla::vsiplugin::AttributeMap attribute_map_;
};

// REGISTER_XLA_OP(Name("ArgMax")
//                     .Device(DEVICE_VSI_NPU_XLA_JIT)
//                     .CompileTimeConstantInput("dimension").TypeConstraint(
//                         "T", {DT_INT32, DT_INT64}), 
//                 ArgMaxOp);

}  // namespace tensorflow
