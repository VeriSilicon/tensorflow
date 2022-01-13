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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_PASSED_INSERTTRANSPOSE_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_PASSED_INSERTTRANSPOSE_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace vsiplugin {

/**
 * This class finds all instructions that explicitly require layout info. For
 * each one of them, it insert a transpose instruction in front of them.
 */
class InsertTranspose : public HloModulePass {
public:
    InsertTranspose() {}
    ~InsertTranspose() = default;
    absl::string_view name() const override { return "insert-transpose"; }
    StatusOr<bool> Run(HloModule* module) override;
private:
    StatusOr<bool> InsertTransposeForConv(HloInstruction *conv);
    HloInstruction* repalceConvWithNew(HloInstruction *oldCconv,
                    HloInstruction *input, HloInstruction *weight);
    HloInstruction* insertTransposeforHlo(HloInstruction *hlo, std::vector<int64_t> &dim_index);
};

}  // namespace vsiplugin
}  // namespace xla
#endif