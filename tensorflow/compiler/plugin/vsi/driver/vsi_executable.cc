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

#include "tensorflow/compiler/plugin/vsi/driver/vsi_executable.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"

namespace xla{
namespace vsiplugin{

VsiExecutable::VsiExecutable(std::shared_ptr<HloModule> hlo_module,
        VsiExecutor *executor) :
        Executable(hlo_module,
                    /*hlo_profile_printer_data=*/nullptr,
                    /*hlo_profile_index_map=*/nullptr),
        visitor_(std::move(std::make_unique<BaseVisitor>(executor))),
        executor_(executor) {
            visitor_->ResetVisitStates();
        }

VsiExecutable::~VsiExecutable()
{
}

// StatusOr<ExecutionOutput> VsiExecutable::ExecuteAsyncOnStream(
//     const ServiceExecutableRunOptions* run_options,
//     std::vector<ExecutionInput> arguments,
//     HloExecutionProfile* hlo_execution_profile) {
//     if (hlo_module_) {
//         const HloComputation* entry_comp = hlo_module_->entry_computation();
//         CHECK_EQ(entry_comp->num_parameters(), arguments.size())
//             << "Wrong number of arguments passed when running executable";
//         for (int64 i = 0; i < entry_comp->num_parameters(); ++i) {
//         const Shape& expected_shape =
//             entry_comp->parameter_instruction(i)->shape();
//         const Shape& actual_shape = arguments[i].Buffers().shape();
//         TF_RET_CHECK(
//             ShapeUtil::DynamicShapeIsCompatible(actual_shape, expected_shape))
//             << "Shape mismatch on argument " << i << ", "
//             << expected_shape.ToString(/*print_layout=*/true) << " vs. "
//             << actual_shape.ToString(/*print_layout=*/true);
//         }
//     }

//     std::vector<se::DeviceMemoryBase> argument_buffers;
//     std::vector<Shape> argument_shapes;
//     for(size_t i=0;i<arguments.size();i++){
//         const se::DeviceMemoryBase& argument_buffer =
//                 arguments[i].Buffer(/*index=*/{}).AsDeviceMemoryBase();
//         const Shape& argument_shape = arguments[i].shape();
//         argument_buffers.push_back(argument_buffer);
//         argument_shapes.push_back(argument_shape);
//     }

//     VLOG(1) << "Execute " << module().name();
//     if (VLOG_IS_ON(1)) {
//         for (const auto& a : argument_buffers) {
//         VLOG(1) << "-- argument " << a.opaque();
//         }
//     }

//     se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();

//         // auto* host_stream = dynamic_cast<se::host::HostStream*>(
//         //     run_options->stream()->implementation());
//         // se::Stream* stream = run_options->stream();
//         // se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();


//     // se::DeviceMemoryBase out = arguments[allocation.parameter_number()]
//     //                                .Buffer(allocation.param_shape_index())
//     //                                .AsDeviceMemoryBase();

// }

StatusOr<ExecutionOutput> VsiExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile){
        mtx_.lock();
        LOG(INFO) << "Execute " << module().name() << " :: " << (void*)this
            << " :: "<< tensorflow::Env::Default()->GetCurrentThreadId();

        se::Stream* stream = run_options->stream();
        se::StreamExecutor* executor = stream->parent();
        const se::Platform* platform = executor->platform();

        // Convert the ShapeTree to a ShapedBuffer. We do this so we can call
        // TransferManager methods below.
        std::vector<ShapedBuffer> argument_buffers;
        argument_buffers.reserve(arguments.size());
        for (auto& argument : arguments) {
            const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
            argument_buffers.push_back(ShapedBuffer(buffers.shape(), buffers.shape(),
                                                    platform,
                                                    executor->device_ordinal()));
            auto in_it = buffers.begin();
            auto out_it = argument_buffers.back().buffers().begin();
            for (; in_it != buffers.end(); ++in_it, ++out_it) {
            out_it->second = in_it->second.AsDeviceMemoryBase();
            }
        }

        const HloComputation* computation = module().entry_computation();
        if (computation->num_parameters() != arguments.size()) {
            return tensorflow::errors::Internal(
            "Mismatch between argument count and graph parameter count.");
        }

        TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                      TransferManager::GetForPlatform(platform));
        // Transform the ShapedBuffer arguments into literals which the evaluator
        // consumes.
        std::vector<Literal> arg_literals;
        for (int64 p = 0; p < computation->num_parameters(); ++p) {
            TF_ASSIGN_OR_RETURN(Literal arg_literal,
                                transfer_manager->TransferLiteralFromDevice(
                                    run_options->stream(), argument_buffers[p]));
            arg_literals.push_back(std::move(arg_literal));
        }
 
        auto tensor = visitor_->evaluate(*computation, arg_literals);

        // Transform the result literal back into a ShapedBuffer.
        auto root_instr = computation->root_instruction();
        const Shape& result_shape = root_instr->shape();

        TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result_buffers,
                            transfer_manager->AllocateScopedShapedBuffer(
                                result_shape, run_options->allocator(),
                                executor->device_ordinal()));


        if(!result_shape.IsTuple()){
            for(auto& pair:result_buffers.buffers()){
                const ShapeIndex& index = pair.first;
                se::DeviceMemoryBase& memory_base = pair.second;
                const Shape& subshape =
                    ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
                std::cout<<"no-tuple  result buffer info "<<subshape.ToString()<<std::endl;
                tensor[0]->CopyDataFromTensor(memory_base.opaque());
            }
        }else{
            int32_t count = 0;
            auto top_shape_memory = result_buffers.buffers();

            se::DeviceMemoryBase top_memory_base;
            for(auto& pair:result_buffers.buffers()){
                if(count == 0){
                    top_memory_base = pair.second;
                    std::cout<<"top_memory_base location is "<<top_memory_base.opaque()<<std::endl;
                    count++;
                }
            }

            count = -1;
            for(auto& pair:result_buffers.buffers()){
                count++;
                if(count == 0) continue;
                const ShapeIndex& index = pair.first;
                se::DeviceMemoryBase& memory_base = pair.second;
                const Shape& subshape =
                    ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
                std::cout<<"tuple result buffer info "<<subshape.ToString()<<std::endl;

                tensor[count - 1]->CopyDataFromTensor(memory_base.opaque());
                *(size_t*)(top_memory_base.opaque() + sizeof(void*)*(count - 1)) = (size_t)memory_base.opaque();
                std::cout<<"sub tensor mem is "<<memory_base.opaque()<<std::endl;
                
            }
        }
        // auto output_size =
        //     ShapeUtil::ByteSizeOf(root_instr->shape(), sizeof(void*));
        // LOG(INFO) << "Output Size:" << output_size;

        // se::DeviceMemoryBase devMem = executor_->Allocate(output_size, 0);
        // tensor[0]->CopyDataFromTensor(devMem.opaque());

        // /*for vsi, memory layout always is dim 0 as major */
        // auto shape = root_instr->shape();
        // *shape.mutable_layout() =
        //     LayoutUtil::GetDefaultLayoutForRank(root_instr->shape().rank());

        //     ScopedShapedBuffer shaped_buffer(shape, shape,
        //         run_options->allocator(), executor->device_ordinal());

        // const ShapeIndex shapeIndex;
        // for (auto& pair : shaped_buffer.buffers()) {
        //     pair.second = devMem;
        // }
        ExecutionOutput result(std::move(result_buffers));
        mtx_.unlock();
        return result;
    }

StatusOr<std::vector<ScopedShapedBuffer>> VsiExecutable::ExecuteOnStreams(
    absl::Span<const ServiceExecutableRunOptions> run_options,
    absl::Span<const absl::Span<const ShapedBuffer* const>> arguments){
        LOG(FATAL)<<"not implement";
    }

Status VsiExecutable::PopulateExecutionProfile(
    ExecutionProfile* execution_profile,
    HloExecutionProfile* hlo_execution_profile, se::Stream* stream){
        LOG(FATAL)<<"not implement";
    }

} // namespace vsiplugin
} // namespace xla