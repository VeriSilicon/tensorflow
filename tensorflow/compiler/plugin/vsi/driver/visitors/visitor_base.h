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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_

#include <string>
#include <unordered_map>
#include <mutex>

#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
namespace xla {
namespace vsiplugin {

/*
 * The base visitor handles all operations that are element-wise.
 * This includes all explicitly element-wise ops, for temprarily, they
 * are implemented by hlo_evaluator, and repalce it with AIM implment
 * step by step. All of these have no element to element dependencies.
 */
class BaseVisitor : public DfsHloVisitor {
 public:
    BaseVisitor(VsiExecutor* executor) : executor_(executor),
    graph_(executor->getContext()->CreateGraph()) {};

    std::shared_ptr<tim::vx::Tensor> createTensorFromTupleShape(const Shape &shape,
        int64 index,tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT){
        tim::vx::ShapeType timShape;
        tim::vx::Quantization timQuant;
        std::cout<<"shape info ";


        auto output_shape = shape.tuple_shapes(index);

        if(output_shape.is_static() && output_shape.has_layout()){
            for( auto d : output_shape.layout().minor_to_major())
              timShape.push_back(output_shape.dimensions(d));
        }

        if(timShape.size() == 0){
          timShape.push_back(1);
        }
        for(uint32_t i=0;i<timShape.size();i++){
          std::cout<<timShape[i]<<" ";
        }
        std::cout<<std::endl;
        auto type = convertTfPrimitiveTypeToTim(output_shape.element_type());
        std::unique_lock<std::mutex> lock(mutex_);
        tim::vx::TensorSpec timSpec(type, timShape,
                    attr, timQuant);
        return graph_->CreateTensor(timSpec);
    }
  
    std::shared_ptr<tim::vx::Tensor> createTensorFromShape(const Shape &shape,
        tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT){
        tim::vx::ShapeType timShape;
        tim::vx::Quantization timQuant;
        std::cout<<"shape info ";
        if(shape.is_static() && shape.has_layout()){
            for( auto d : shape.layout().minor_to_major())
              timShape.push_back(shape.dimensions(d));
        }

        if(timShape.size() == 0){
          timShape.push_back(1);
        }
        for(uint32_t i=0;i<timShape.size();i++){
          std::cout<<timShape[i]<<" ";
        }
        std::cout<<std::endl;
        auto type = convertTfPrimitiveTypeToTim(shape.element_type());
        std::unique_lock<std::mutex> lock(mutex_);
        tim::vx::TensorSpec timSpec(type, timShape,
                    attr, timQuant);
        return graph_->CreateTensor(timSpec);
    }

    std::shared_ptr<tim::vx::Tensor> createTensorFromShape(tim::vx::DataType dataType,
        std::vector<uint32_t> shape,
        tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT){
        tim::vx::ShapeType timShape;
        tim::vx::Quantization timQuant;
        for( auto d : shape)
          timShape.push_back(d);
        std::cout<<"shape info ";
        if(timShape.size() == 0){
          timShape.push_back(1);
        }
        for(uint32_t i=0;i<timShape.size();i++){
          std::cout<<timShape[i]<<" ";
        }
        std::cout<<std::endl;

        std::unique_lock<std::mutex> lock(mutex_);
        tim::vx::TensorSpec timSpec(dataType, timShape,
                    attr, timQuant);
        return graph_->CreateTensor(timSpec);
    }

  static tim::vx::DataType convertTfPrimitiveTypeToTim(xla::PrimitiveType xlaType){
    LOG(INFO) << "convertTfPrimitiveTypeToTim: xlaType: " << xlaType <<std::endl;
      switch(xlaType){
        case PRED:{
          return tim::vx::DataType::BOOL8;
        }
        case S64:{
          return tim::vx::DataType::INT32;
        }
        case S8:{
          return tim::vx::DataType::INT8;
        }
        case U8:{
          return tim::vx::DataType::UINT8;
        }
        case S16:{
          return tim::vx::DataType::INT16;
        }
        case U16:{
          return tim::vx::DataType::UINT16;
        }
        case S32:{
          return tim::vx::DataType::INT32;
        }
        case U32:{
          return tim::vx::DataType::UINT32;
        }
        case F32:{
          return tim::vx::DataType::FLOAT32;
        }
        case BF16:{
          return tim::vx::DataType::FLOAT16;
        }
        case F16:{
          return tim::vx::DataType::FLOAT16;
        }
        case F64:{
          return tim::vx::DataType::FLOAT32;
        }
        default:
          LOG(FATAL)<<"not supported datat type";
      }
  }

  /*dim_index: store the demension index info of the $hlo$ as order major_to_minor: {N, C, ..... }
    if it should be inserted a transpose, its output would be returned.*/
  std::shared_ptr<tim::vx::Tensor> insertTranspose(const HloInstruction *hlo, std::vector<uint32_t> &dim_index);

  virtual const Shape& GetOutputShape(HloInstruction*) const;

    Literal evaluate(const HloComputation& computation
         /*absl::Span<const Literal* const> arg_literals*/);
    
    std::vector<std::shared_ptr<tim::vx::Tensor>> evaluate(const HloComputation& computation,
        std::vector<Literal>& argument_literals);

    Status HandleHloOp(HloInstruction* hlo);

    Status FinishVisit(HloInstruction* root) final;

    // Returns the already-evaluated literal result for the instruction.
    //
    // A Constant instruction is considered evaluated and its literal will be
    // returned directly without looking up the cache.
    //
    // Similarly, a Parameter instruction is considered evaluated and its literal
    // is looked up in arg_literals.
    //
    // Crash with log if the given instruction has not been evaluated previously.
    const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
        if (hlo->IsConstant()) {
            return hlo->literal();
        }
        // if (hlo->opcode() == HloOpcode::kParameter) {
        //     return *arg_literals_.at(hlo->parameter_number());
        // }
        auto it = evaluated_.find(hlo);
        CHECK(it != evaluated_.end())
            << "could not find evaluated value for: " << hlo->ToString();
        return it->second;
    }

    const std::vector<std::shared_ptr<tim::vx::Tensor>> GetEvaluatedTensorFor(const HloInstruction* hlo) {
        //return createTensorFromShape(hlo->shape());
        auto it = kVsiRunTensorContainer_.find(hlo);
        CHECK(it != kVsiRunTensorContainer_.end())
            << "could not find evaluated value for: " << hlo->ToString();
        return kVsiRunTensorContainer_[hlo];
    }

  // Called by HandleElementwiseBinarythe FinishVisit.
  virtual Status FinishScopedVisit(HloInstruction* root) {
    return Status::OK();
  }
  Status HandleElementwiseBinary(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;

  Status HandleConstant(HloInstruction* hlo) override;

  Status HandleParameter(HloInstruction* hlo) override;

  Status HandleReshape(HloInstruction* hlo) override;

  Status HandleTranspose(HloInstruction* hlo) override;

  Status HandleTuple(HloInstruction* hlo) override;

  Status HandleGetTupleElement(HloInstruction* hlo) override;

  Status HandleConvolution(HloInstruction* hlo) override;

  Status HandleReverse(HloInstruction* hlo) override;

  Status HandleConvert(HloInstruction* hlo) override;

  //Status HandleSlice(HloInstruction* hlo) override;

  Status HandleBroadcast(HloInstruction* hlo) override;

  Status HandleConcatenate(HloInstruction* hlo) override;
  
  Status HandleCompare(HloInstruction* hlo) override;

  Status HandleSelect(HloInstruction* hlo) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleDot(HloInstruction* hlo) override;

  Status HandleIota(HloInstruction* hlo) override;

#define HANDLE_AS_HLO_OP(Name) \
  Status Name(HloInstruction* inst) override { return HandleHloOp(inst); }

  /*
   * Operations not processed by this visitor.
   */
#define UNIMPLEMENTED(Name) \
  Status Name(HloInstruction* inst) override { \
    LOG(INFO)<< "@@ unimplement instruction "<<__FUNCTION__; \
    return Unimplemented(inst); \
    };

  UNIMPLEMENTED(HandleTupleSelect)
  //UNIMPLEMENTED(HandleConvert)
  UNIMPLEMENTED(HandleCollectivePermuteStart)
  UNIMPLEMENTED(HandleCollectivePermuteDone)
  UNIMPLEMENTED(HandleRngBitGenerator)
  UNIMPLEMENTED(HandleBitcastConvert)
  UNIMPLEMENTED(HandleAllReduce)
  UNIMPLEMENTED(HandleAllGather)
  UNIMPLEMENTED(HandleFusion)
  UNIMPLEMENTED(HandleCall)
  UNIMPLEMENTED(HandleCustomCall)
  UNIMPLEMENTED(HandleMap)
  UNIMPLEMENTED(HandleConditional)
  UNIMPLEMENTED(HandleInfeed)
  UNIMPLEMENTED(HandleAfterAll)
  UNIMPLEMENTED(HandleReal)
  UNIMPLEMENTED(HandleAllToAll)
  UNIMPLEMENTED(HandleAddDependency)
  //UNIMPLEMENTED(HandleElementwiseUnary)
  UNIMPLEMENTED(HandleClamp)
  //UNIMPLEMENTED(HandleSelect)
  //UNIMPLEMENTED(HandleCompare)
  UNIMPLEMENTED(HandleRng)
  UNIMPLEMENTED(HandleSlice)
  UNIMPLEMENTED(HandleDynamicSlice)
  UNIMPLEMENTED(HandleDynamicUpdateSlice)
  UNIMPLEMENTED(HandleSelectAndScatter)
  UNIMPLEMENTED(HandleWhile)
  UNIMPLEMENTED(HandlePad)
  UNIMPLEMENTED(HandleSort)
  //UNIMPLEMENTED(HandleReduce)
  UNIMPLEMENTED(HandleBitcast)
  //UNIMPLEMENTED(HandleBroadcast)
  UNIMPLEMENTED(HandleReducePrecision)
  UNIMPLEMENTED(HandleOutfeed)
  UNIMPLEMENTED(HandleSend)
  UNIMPLEMENTED(HandleSendDone)
  UNIMPLEMENTED(HandleRecv)
  UNIMPLEMENTED(HandleRecvDone)
  UNIMPLEMENTED(HandleBatchNormInference)
  UNIMPLEMENTED(HandleBatchNormTraining)
  UNIMPLEMENTED(HandleBatchNormGrad)
  UNIMPLEMENTED(HandleFft)
  UNIMPLEMENTED(HandleGather)
  UNIMPLEMENTED(HandleCopy)
  //UNIMPLEMENTED(HandleIota)
  UNIMPLEMENTED(HandleScatter)
  UNIMPLEMENTED(HandleCollectivePermute)
  //UNIMPLEMENTED(HandleConcatenate)
  UNIMPLEMENTED(HandleGetDimensionSize)
  UNIMPLEMENTED(HandleReplicaId)
  UNIMPLEMENTED(HandleTriangularSolve)
  UNIMPLEMENTED(HandleCholesky)
  UNIMPLEMENTED(HandlePartitionId)
  UNIMPLEMENTED(HandleRngGetAndUpdateState)
  UNIMPLEMENTED(HandleCopyStart)
  UNIMPLEMENTED(HandleCopyDone)
  UNIMPLEMENTED(HandleSetDimensionSize)
  //UNIMPLEMENTED(HandleDot)
  UNIMPLEMENTED(HandleReduceWindow)

 protected:
  Status Unimplemented(HloInstruction* inst);

  const std::string name_ = "vsi base visitor";

  std::unique_ptr<HloEvaluator> cpu_evaluator_;

private:
    VsiExecutor *executor_;

    // Tracks the HLO instruction and its evaluated literal result.
    // Parameters and constants aren't stored here,
    // TODO: it is better the Literal value was repalced with device memory
    //       handle.
    std::mutex mutex_;
    std::unordered_map<const HloInstruction *, Literal> evaluated_ TF_GUARDED_BY(mutex_);
    std::unordered_map<const HloInstruction*, std::vector<std::shared_ptr<tim::vx::Tensor>>>
        kVsiRunTensorContainer_ TF_GUARDED_BY(mutex_);
    // std::unordered_map<const HloInstruction*, std::shared_ptr<tim::vx::Tensor>>
    //     kVsiRunTensorContainer_ TF_GUARDED_BY(mutex_);
    std::vector<Literal> arg_literals_;
    std::shared_ptr<tim::vx::Graph> graph_;
};

}  // namespace vsiplugin
}  // namespace xla

#endif
