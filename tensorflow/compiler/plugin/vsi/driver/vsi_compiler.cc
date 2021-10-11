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

#include "tensorflow/compiler/plugin/vsi/driver/vsi_compiler.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/compiler/plugin/vsi/driver/passes/InsertTranspose.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_platform.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_platform_id.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executable.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"

namespace xla {
namespace vsiplugin {

StatusOr<std::unique_ptr<HloModule>> VsiCompiler::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
    HloPassPipeline pipeline("vsi-npu-pass");
    pipeline.AddPass<InsertTranspose>();
    pipeline.Run(hlo_module.get());

    return std::move(hlo_module);
}

StatusOr<std::unique_ptr<Executable>> VsiCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
    TF_RET_CHECK(stream_exec != nullptr);

    VLOG(1) << "Run backend " << hlo_module->name();
    // Create executable from only the Hlo module.
    std::unique_ptr<Executable> executable =
        absl::make_unique<vsiplugin::VsiExecutable>(
            std::move(hlo_module),
            dynamic_cast<VsiExecutor*>(stream_exec->implementation()));

    return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> VsiCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
    if (module_group->empty()) {
        return std::vector<std::unique_ptr<Executable>>();
    }
    if (module_group->size() > 1) {
        return tensorflow::errors::Unimplemented(
            "Compilation of multiple HLO modules is not supported on Interpreter.");
    }
    if (stream_exec.size() != 1 || stream_exec[0].size() != 1) {
        return tensorflow::errors::Unimplemented(
            "Unexpected number of StreamExecutor's.");
    }
    auto hlo_modules = module_group->ConsumeModules();
    LOG(INFO) << "module name: " << hlo_modules[0]->name();
    TF_ASSIGN_OR_RETURN(auto module,
                        RunHloPasses(std::move(hlo_modules[0]), stream_exec[0][0],
                                    device_allocator));
    TF_ASSIGN_OR_RETURN(
        auto executable,
        RunBackend(std::move(module), stream_exec[0][0], device_allocator));
    std::vector<std::unique_ptr<Executable>> ret;
    ret.push_back(std::move(executable));
    return std::move(ret);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
VsiCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                    const AotCompilationOptions& aot_options) {}

HloCostAnalysis::ShapeSizeFunction VsiCompiler::ShapeSizeBytesFunction() const {
    return [](const Shape& shape) -> int64_t{
        if (shape.IsOpaque()) {
            return sizeof(void*);
        }
        return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
    };
}

se::Platform::Id VsiCompiler::PlatformId() const { return xla::vsiplugin::kVsiPlatformId; }

} // namespace vsiplugin
} // namespace xla

static bool InitModule() {
    xla::Compiler::RegisterCompilerFactory(
        xla::vsiplugin::kVsiPlatformId,
        []() { return absl::make_unique<xla::vsiplugin::VsiCompiler>(); });

    xla::ComputationPlacer::RegisterComputationPlacer(
        xla::vsiplugin::kVsiPlatformId,
        []() { return absl::make_unique<xla::ComputationPlacer>(); });
    return true;
}
static bool module_initialized = InitModule();
