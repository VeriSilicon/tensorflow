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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VSI_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VSI_EXECUTABLE_H_

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/plugin/vsi/driver/visitors/visitor_base.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
namespace se = stream_executor;

namespace xla{
namespace vsiplugin{

class VsiExecutable : public Executable  {
public:
    explicit VsiExecutable( std::shared_ptr<HloModule> hlo_module,
    VsiExecutor *executor);

    ~VsiExecutable();

    StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
        const ServiceExecutableRunOptions* run_options,
        std::vector<ExecutionInput> arguments,
        HloExecutionProfile* hlo_execution_profile) override ;

    // Same as ExecuteOnStream(), but runs this executable on multiple
    // streams. arguments[i] contains the arguments to the execution on
    // run_options[i]->stream() and the returned value is at index i of the
    // returned vector.
    StatusOr<std::vector<ScopedShapedBuffer>> ExecuteOnStreams(
        absl::Span<const ServiceExecutableRunOptions> run_options,
        absl::Span<const absl::Span<const ShapedBuffer* const>> arguments) override;

    // Populates `hlo_execution_profile` from `executor`. This is implicit in any
    // Execute* API call that takes a hlo_execution_profile argument, but must be
    // called explicitly for other (async, for example) variants after the stream
    // has completed.
    Status PopulateExecutionProfile(
        ExecutionProfile* execution_profile,
        HloExecutionProfile* hlo_execution_profile, se::Stream* stream) override;

private:
    std::unique_ptr<BaseVisitor> visitor_;
    VsiExecutor *executor_;
};

} // namespace vsiplugin
} // namespace xla
#endif