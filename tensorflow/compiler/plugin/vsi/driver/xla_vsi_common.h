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
#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_XLA_VSI_COMMON_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_XLA_VSI_COMMON_H_

namespace tensorflow {

const char* const DEVICE_XLA_VSI_NPU = "NPU";
const char* const DEVICE_VSI_NPU_XLA_JIT = "XLA_NPU_JIT";
const char* const PLATFORM_NAME = "vsi-npu";

}  // namespace tensorflow

#endif
