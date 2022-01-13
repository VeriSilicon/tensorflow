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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VSI_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VSI_UTILS_H_

#include "tensorflow/core/platform/default/integral_types.h"
namespace xla{

namespace vsiplugin{

using int64 = tensorflow::int64;
using int8  = tensorflow::int8;
using int16 = tensorflow::int16;
using int32 = tensorflow::int32;
using int64 = tensorflow::int64;
using uint8 = tensorflow::uint8;
using uint16 = tensorflow::uint16;
using uint32 = tensorflow::uint32;
using uint64 = tensorflow::uint64;

template <typename To, typename From>
To convert_array(const From& from) {
  To out;
  for (const auto& e : from) {
    out.push_back(e);
  }
  return out;
}

} // namespace vsiplugin
} // namespace xla
#endif