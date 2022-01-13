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

#include "tensorflow/compiler/plugin/vsi/driver/passes/InsertTranspose.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {
namespace vsiplugin {

StatusOr<bool> InsertTranspose::Run(HloModule* module){
    bool changed = false;
    XLA_VLOG_LINES(2, "InsertTranspose::Run(), before:\n" + module->ToString());

    for (HloComputation* computation : module->MakeComputationPostOrder()) {
        for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
            if (instruction->opcode() == HloOpcode::kConvolution) {
                changed |= InsertTransposeForConv(instruction).ValueOrDie();
            }
        }
    }
    XLA_VLOG_LINES(2,
            "InsertTranspose::Run(), after:\n" + module->ToString());
    return changed;
}

StatusOr<bool> InsertTranspose::InsertTransposeForConv(HloInstruction *conv){
    auto lhs = conv->operand(0);
    auto rhs = conv->operand(1);
    const auto& window = conv->window();
    const Shape& result_shape = conv->shape();
    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(lhs_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(rhs_shape));
    CHECK(lhs_shape.IsArray());
    CHECK(rhs_shape.IsArray());
    CHECK(ShapeUtil::SameElementType(lhs_shape, rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, result_shape));

    const auto& dnums = conv->convolution_dimension_numbers();
    const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
    CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, 2); /*vsi requirement*/
    CHECK_GE(num_spatial_dims, 0);
    CHECK_EQ(window.dimensions_size(), num_spatial_dims);

    const auto lhs_rank = lhs_shape.rank();
    const auto rhs_rank = rhs_shape.rank();
    CHECK_EQ(num_spatial_dims + 2, lhs_rank);
    CHECK_EQ(num_spatial_dims + 2, rhs_rank);

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferConvolveShape(
                            lhs_shape, rhs_shape, conv->feature_group_count(),
                            conv->batch_group_count(), window, dnums, absl::nullopt));
    CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    std::vector<int64_t> input_dim;
    input_dim.push_back(dnums.input_batch_dimension());
    input_dim.push_back(dnums.input_feature_dimension());

    std::vector<int64_t> weight_dim;
    weight_dim.push_back(dnums.kernel_output_feature_dimension());
    weight_dim.push_back(dnums.kernel_input_feature_dimension());
    for(size_t i = 2; i < lhs_rank; i++){
        input_dim.push_back(dnums.input_spatial_dimensions(i - 2));
        weight_dim.push_back(dnums.kernel_spatial_dimensions(i - 2));
    }

    auto input_instr = insertTransposeforHlo(conv->mutable_operand(0), input_dim);
    auto weight_instr = insertTransposeforHlo(conv->mutable_operand(1), weight_dim);
    auto newConv = repalceConvWithNew(conv, input_instr, weight_instr);

    return newConv != nullptr;
}

HloInstruction* InsertTranspose::repalceConvWithNew(HloInstruction *oldConv,
                    HloInstruction *input, HloInstruction *weight){
    ConvolutionDimensionNumbers dim_numbers = oldConv->convolution_dimension_numbers();
    /*the layout of the new instruction created by insertTransposeforHlo, is:
    ** input:  {N C H W ....}
    ** weight: {O I H W ...}*/

    auto input_shape = input->shape();
    auto input_layout_minor_to_major = input_shape.layout().minor_to_major();
    dim_numbers.set_input_batch_dimension(
        input_layout_minor_to_major.back());
    dim_numbers.set_input_feature_dimension(
        input_layout_minor_to_major[input_layout_minor_to_major.size()-2] );

    dim_numbers.clear_input_spatial_dimensions();
    for(auto i = 2; i < input_layout_minor_to_major.size(); i++){
        dim_numbers.add_input_spatial_dimensions(
            input_layout_minor_to_major[input_layout_minor_to_major.size() - i - 1]);
    }

    auto weight_layout_minor_to_major = weight->shape().layout().minor_to_major();
    dim_numbers.set_kernel_input_feature_dimension(
        weight_layout_minor_to_major[weight_layout_minor_to_major.size()-2]
    );
    dim_numbers.set_kernel_output_feature_dimension(
        weight_layout_minor_to_major.back()
    );
    dim_numbers.clear_kernel_spatial_dimensions();
    for(auto i = 2; i < weight_layout_minor_to_major.size(); i++){
        dim_numbers.add_kernel_spatial_dimensions(
            weight_layout_minor_to_major[weight_layout_minor_to_major.size() - i - 1]
        );
    }
    
    auto new_convolution = HloInstruction::CreateConvolve(
          oldConv->shape(), input, weight,
          oldConv->feature_group_count(),
          oldConv->batch_group_count(),
          oldConv->window(),
          dim_numbers, oldConv->precision_config());

    oldConv->parent()->ReplaceWithNewInstruction(
          oldConv, std::move(new_convolution));

    return nullptr;
}

HloInstruction* InsertTranspose::insertTransposeforHlo(HloInstruction *hlo, std::vector<int64_t> &dim_index){
    CHECK_EQ(hlo->shape().dimensions().size(), dim_index.size());
    std::vector<int64_t> new_output_dimensions;
    for(auto d : dim_index){
        new_output_dimensions.push_back(hlo->shape().dimensions(d));
    }

    auto parent = hlo->parent();
    Shape shape = ShapeUtil::MakeShape(hlo->shape().element_type(), new_output_dimensions);
    return  parent->AddInstruction(HloInstruction::CreateTranspose(
        shape, hlo, dim_index));

    // size_t dim_size = dim_index.size();
    // std::vector<int64_t> output_dims(dim_size, 1);
    // std::vector<uint32_t> perm(dim_size, 1);

    // auto input_tensor = GetEvaluatedTensorFor(hlo);

    // /*check if the shape is {WHCN} , if not, a transpose would be inserted to covert the layout. */
    // bool is_need_insert_transpose = false;
    // for(int i = 0; i < dim_size; i++){
    //     if(dim_index[i] == shape.layout().minor_to_major()[dim_size - i - 1]){
    //         perm[dim_size - 1 - i] = dim_size - i - 1;
    //     }else{
    //         is_need_insert_transpose = true;
    //         for(int j = 0; j < dim_size; j++){
    //             if(dim_index[i] != shape.layout().minor_to_major()[j])
    //                 continue;
    //             perm[dim_size - 1 - i] = j;
    //             break;
    //         }
    //     }
    // }
    // if(is_need_insert_transpose){
    //     auto input_shape = input_tensor->GetShape();
    //     std::vector<uint32_t> output_shape;
    //     for(auto d : perm){
    //         output_shape.push_back(input_shape[d]);
    //     }
    //     auto output_tensor = createTensorFromShape(
    //         convertTfPrimitiveTypeToTim(hlo->shape().element_type()), output_shape,
    //         tim::vx::TensorAttribute::OUTPUT);
    //     auto transposeOp = graph_->CreateOperation<tim::vx::ops::Transpose>(perm);
    //     transposeOp->BindInput(input_tensor).BindOutput(output_tensor);

    //     return output_tensor;
    // }
    // return input_tensor;
}

}  // namespace vsiplugin
}  // namespace xla
