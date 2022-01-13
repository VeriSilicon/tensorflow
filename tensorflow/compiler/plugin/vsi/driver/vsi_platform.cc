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

#include "tensorflow/compiler/plugin/vsi/driver/vsi_platform.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"

#include "tim/vx/context.h"

namespace xla{
namespace vsiplugin{

VsiPlatform::VsiPlatform() {
  kVsiContext = tim::vx::Context::Create();
}

VsiPlatform::~VsiPlatform() {}

se::Platform::Id VsiPlatform::id() const { return id_; }

int VsiPlatform::VisibleDeviceCount() const { return 1; }

const std::string& VsiPlatform::Name() const { return name_; }

port::StatusOr<std::unique_ptr<se::DeviceDescription>>
VsiPlatform::DescriptionForDevice(int ordinal) const {
  
  // todo: open it when executor finish.
  //return VsiExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<se::StreamExecutor*> VsiPlatform::ExecutorForDevice(
    int ordinal) {
  se::StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = se::PluginConfig();
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<se::StreamExecutor*>
VsiPlatform::ExecutorForDeviceWithPluginConfig(
    int ordinal, const se::PluginConfig& conplugin_configfig){
  se::StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = se::PluginConfig();
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<se::StreamExecutor*> VsiPlatform::GetExecutor(
    const se::StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<se::StreamExecutor>>
VsiPlatform::GetUncachedExecutor(
    const se::StreamExecutorConfig& config) {

// LOG(FATAL) << "not yet implemented: register executor trace listener";

// TODO: open it when to finish implement of VsiExecutor
  auto executor = absl::make_unique<se::StreamExecutor>(
      this, absl::make_unique<VsiExecutor>(kVsiContext, config.ordinal, config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

void VsiPlatform::RegisterTraceListener(
    std::unique_ptr<se::TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register executor trace listener";
}

void VsiPlatform::UnregisterTraceListener(se::TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister executor trace listener";
}

static void InitializeVsiPlatform() {
 auto status = se::MultiPlatformManager::PlatformWithName("vsi-npu");
 if (!status.ok()) {

    std::unique_ptr<se::Platform> platform(new VsiPlatform);
    SE_CHECK_OK(se::MultiPlatformManager::RegisterPlatform(std::move(platform)));
 }
}
} // namespace vsiplugin
} // namespace xla

REGISTER_MODULE_INITIALIZER(
    vsi_platform,
    xla::vsiplugin::InitializeVsiPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
DECLARE_MODULE_INITIALIZER(multi_platform_manager);
DECLARE_MODULE_INITIALIZER(multi_platform_manager_listener);
REGISTER_MODULE_INITIALIZER_SEQUENCE(vsi_platform,
                                     multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                      vsi_platform);