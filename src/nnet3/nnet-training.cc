// nnet3/nnet-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3/nnet-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetTrainer::NnetTrainer(const NnetTrainerOptions &config,
                         Nnet *nnet):
    config_(config),
    nnet_(nnet),
    compiler_(*nnet, config_.optimize_config),
    num_minibatches_processed_(0),
    state_preserving_training_(false) {
  if (config.zero_component_stats)
    ZeroComponentStats(nnet);
  if (config.momentum == 0.0 && config.max_param_change == 0.0) {
    delta_nnet_= NULL;
  } else {
    KALDI_ASSERT(config.momentum >= 0.0 &&
                 config.max_param_change >= 0.0);
    delta_nnet_ = nnet_->Copy();
    bool is_gradient = false;  // setting this to true would disable the
                               // natural-gradient updates.
    SetZero(is_gradient, delta_nnet_);
  }
  if (config.minibatch_chunk_size > 0)
    state_preserving_training_ = true;
}


void NnetTrainer::Train(const NnetExample &eg) {
  if (!state_preserving_training_) {
    bool need_model_derivative = true;
    ComputationRequest request;
    GetComputationRequest(*nnet_, eg, need_model_derivative,
                          config_.store_component_stats,
                          &request);
    const NnetComputation *computation = compiler_.Compile(request);

    NnetComputer computer(config_.compute_config, *computation,
                          *nnet_,
                          (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
    // give the inputs to the computer object.
    computer.AcceptInputs(*nnet_, eg.io);
    computer.Forward();

    this->ProcessOutputs(eg, &computer);
    computer.Backward();
  } else {
    int32 num_chunks = -1, left_context = -1, right_context = -1;
    ComputeSimpleNnetContext(*nnet_, &left_context, &right_context); //TODO: avoid call it for every minibatch
    std::vector<NnetExample> splitted;
    eg.SplitChunk(config_.minibatch_chunk_size, left_context, right_context,
		  &num_chunks, &splitted);

    std::vector<std::string> recurrent_output_names;
    GetRecurrentOutputNames(&recurrent_output_names); //TODO: avoid calling it every minibatch

    std::vector<std::pair<std::string, Matrix<BaseFloat> > > r; 

    std::vector<NnetExample>::iterator iter = splitted.begin();
    for (; iter != splitted.end(); ++iter) {
      // add zero matrices in r as additional inputs for the first minibatch
      if (iter == splitted.begin()) {
        r.reserve(recurrent_output_names.size());
	for (int32 i = 0; i < static_cast<int32>(recurrent_output_names.size());
	     i++) {
	  Matrix<BaseFloat> zero_matrix = Matrix<BaseFloat>(num_chunks,
			  nnet_->OutputDim(recurrent_output_names[i]));
          r.push_back(std::make_pair(recurrent_output_names[i], zero_matrix));
	}
      }

      for (int32 i = 0; i < static_cast<int32>(r.size()); i++) {
	// add to NnetIo the recurrent connections from the previous minibatch
	// as additional inputs
	iter->io.push_back(NnetIo(r[i].first + "_STATE_PREVIOUS_MINIBATCH", 0,
			   r[i].second));
	// correct the indexes: swap indexes "n" and "t" so that 
	// n ranges from 0 to feats.NumRows() - 1 and t is always 0
	std::vector<Index> &indexes = iter->io.back().indexes;
	for (int32 j = 0; j < static_cast<int32>(indexes.size()); j++)
	  std::swap(indexes[j].n, indexes[j].t);

	// add to NnetIo the recurrent connections in the current minibatch 
	// as additional outputs. the output matrix is simply all-zero since
	// we only need its size info for NnetIo::indexes
	iter->io.push_back(NnetIo(r[i].first, 0,
		           r[i].second));
        // correct the indexes.
	indexes = iter->io.back().indexes;
	for (int32 n = 0, j = 0; n < num_chunks; n++)
	  for (int32 t = 0; t < config_.minibatch_chunk_size; t++, j++) {
	    indexes[j].n = n;
	    indexes[j].t = t;
	  }
      }
     
      bool need_model_derivative = true; 
      ComputationRequest request;
      GetComputationRequest(*nnet_, *iter, need_model_derivative,
                            config_.store_component_stats,
                            &request);
      const NnetComputation *computation = compiler_.Compile(request);

      NnetComputer computer(config_.compute_config, *computation,
	                    *nnet_,
                            (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
      // give the inputs to the computer object.
      computer.AcceptInputs(*nnet_, (*iter).io);
      computer.Forward();
     
      this->ProcessOutputs(*iter, &computer);
      computer.Backward();

      // add the recurrent outputs in r as additional inputs
      // for the next minibatch
      GetRecurrentOutputs(config_.minibatch_chunk_size, num_chunks,
		          computer, recurrent_output_names, &r);
    }
  }

  if (delta_nnet_ != NULL) {
    BaseFloat scale = (1.0 - config_.momentum);
    if (config_.max_param_change != 0.0) {
      BaseFloat param_delta =
          std::sqrt(DotProduct(*delta_nnet_, *delta_nnet_)) * scale;
      if (param_delta > config_.max_param_change) {
        if (param_delta - param_delta != 0.0) {
          KALDI_WARN << "Infinite parameter change, will not apply.";
          SetZero(false, delta_nnet_);
        } else {
          scale *= config_.max_param_change / param_delta;
          KALDI_LOG << "Parameter change too big: " << param_delta << " > "
                    << "--max-param-change=" << config_.max_param_change
                    << ", scaling by " << config_.max_param_change / param_delta;
        }
      }
    }
    AddNnet(*delta_nnet_, scale, nnet_);
    ScaleNnet(config_.momentum, delta_nnet_);
  }
}

void NnetTrainer::ProcessOutputs(const NnetExample &eg,
                                 NnetComputer *computer) {
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_->GetNodeIndex(io.name);
    KALDI_ASSERT(node_index >= 0);
    if (nnet_->IsOutputNode(node_index)) {
      ObjectiveType obj_type = nnet_->GetNode(node_index).u.objective_type;
      BaseFloat tot_weight, tot_objf;
      bool supply_deriv = true;
      ComputeObjectiveFunction(io.features, obj_type, io.name,
                               supply_deriv, computer,
                               &tot_weight, &tot_objf);
      objf_info_[io.name].UpdateStats(io.name, config_.print_interval,
                                      num_minibatches_processed_++,
                                      tot_weight, tot_objf);
    }
  }
}

void NnetTrainer::GetRecurrentOutputNames(std::vector<std::string>
		                          *recurrent_output_names) {
  // We assume all output nodes except the one named "output" are 
  // recurrent connections.
  recurrent_output_names->clear();
  for (int32 i = 0; i < static_cast<int32>(nnet_->NumNodes()); i++)
    if (nnet_->IsOutputNode(i) && nnet_->GetNodeName(i) != "output")
      recurrent_output_names->push_back(nnet_->GetNodeName(i));
}

void NnetTrainer::GetRecurrentOutputs(int32 chunk_size,
		                      int32 num_chunks,
		                      NnetComputer &computer,
				      std::vector<std::string>
				      &recurrent_output_names,
		                      std::vector<std::pair<std::string,
				      Matrix<BaseFloat> > > *r) {
  r->clear();
  r->reserve(recurrent_output_names.size());

  for (int32 i = 0; i < static_cast<int32>(recurrent_output_names.size()); i++) {
    // get the cuda matrix correspoding to the recurrent output
    const CuMatrixBase<BaseFloat> &r_cuda_all 
	    = computer.GetOutput(recurrent_output_names[i]);
    KALDI_ASSERT(r_cuda_all.NumRows() == num_chunks * chunk_size); //TODO: need to confirm if it is still the case when chunk_left_context > 0

    // only copy the rows corresponding to the recurrent output of the
    // last frame of each chunk in the previous minibatch
    std::vector<int32> indexes(num_chunks);
    for (int32 j = 0; j < num_chunks; j++)
      indexes[j] = j * chunk_size + chunk_size - 1;
    CuArray<int32> indexes_cuda(indexes);

    CuMatrix<BaseFloat> r_cuda(num_chunks, r_cuda_all.NumCols());
    r_cuda.CopyRows(r_cuda_all, indexes_cuda);

    // copy to (node_name : matrix) pair
    r->push_back(std::make_pair(recurrent_output_names[i],
                                Matrix<BaseFloat>(num_chunks,
                                r_cuda_all.NumCols())));
    r->back().second.CopyFromMat(r_cuda);
  }
}

bool NnetTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;
    ans = ans || info.PrintTotalStats(name);
  }
  return ans;
}

void ObjectiveFunctionInfo::UpdateStats(
    const std::string &output_name,
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    BaseFloat this_minibatch_weight,
    BaseFloat this_minibatch_tot_objf) {
  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase == current_phase + 1); // or doesn't really make sense.
    PrintStatsForThisPhase(output_name, minibatches_per_phase);
    current_phase = phase;
    tot_weight_this_phase = 0.0;
    tot_objf_this_phase = 0.0;
  }
  tot_weight_this_phase += this_minibatch_weight;
  tot_objf_this_phase += this_minibatch_tot_objf;
  tot_weight += this_minibatch_weight;
  tot_objf += this_minibatch_tot_objf;
}

void ObjectiveFunctionInfo::PrintStatsForThisPhase(
    const std::string &output_name,
    int32 minibatches_per_phase) const {
  int32 start_minibatch = current_phase * minibatches_per_phase,
      end_minibatch = start_minibatch + minibatches_per_phase - 1;
  KALDI_LOG << "Average objective function for '" << output_name
            << "' for minibatches " << start_minibatch
            << '-' << end_minibatch << " is "
            << (tot_objf_this_phase / tot_weight_this_phase) << " over "
            << tot_weight_this_phase << " frames.";
}

bool ObjectiveFunctionInfo::PrintTotalStats(const std::string &name) const {
  KALDI_LOG << "Overall average objective function for '" << name << "' is "
            << (tot_objf / tot_weight) << " over " << tot_weight << " frames.";
  KALDI_LOG << "[this line is to be parsed by a script:] "
            << "log-prob-per-frame="
            << (tot_objf / tot_weight);
  return (tot_weight != 0.0);
}

NnetTrainer::~NnetTrainer() {
  delete delta_nnet_;
}

void ComputeObjectiveFunction(const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf) {
  const CuMatrixBase<BaseFloat> &output = computer->GetOutput(output_name);

  if (output.NumCols() != supervision.NumCols())
    KALDI_ERR << "Nnet versus example output dimension (num-classes) "
              << "mismatch for '" << output_name << "': " << output.NumCols()
              << " (nnet) vs. " << supervision.NumCols() << " (egs)\n";

  switch (objective_type) {
    case kLinear: {
      // objective is x * y.
      switch (supervision.Type()) {
        case kSparseMatrix: {
          const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();
          CuSparseMatrix<BaseFloat> cu_post(post);
          // The cross-entropy objective is computed by a simple dot product,
          // because after the LogSoftmaxLayer, the output is already in the form
          // of log-likelihoods that are normalized to sum to one.
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatSmat(output, cu_post, kTrans);
          if (supply_deriv) {
            CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols(),
                                             kUndefined);
            cu_post.CopyToMat(&output_deriv);
            computer->AcceptOutputDeriv(output_name, &output_deriv);
          }
          break;
        }
        case kFullMatrix: {
          // there is a redundant matrix copy in here if we're not using a GPU
          // but we don't anticipate this code branch being used in many cases.
          CuMatrix<BaseFloat> cu_post(supervision.GetFullMatrix());
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptOutputDeriv(output_name, &cu_post);
          break;
        }
        case kCompressedMatrix: {
          Matrix<BaseFloat> post;
          supervision.GetMatrix(&post);
          CuMatrix<BaseFloat> cu_post;
          cu_post.Swap(&post);
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptOutputDeriv(output_name, &cu_post);
          break;
        }
      }
      break;
    }
    case kQuadratic: {
      // objective is -0.5 (x - y)^2
      CuMatrix<BaseFloat> diff(supervision.NumRows(),
                               supervision.NumCols(),
                               kUndefined);
      diff.CopyFromGeneralMat(supervision);
      diff.AddMat(-1.0, output);
      *tot_weight = diff.NumRows();
      *tot_objf = -0.5 * TraceMatMat(diff, diff, kTrans);
      if (supply_deriv)
        computer->AcceptOutputDeriv(output_name, &diff);
      break;
    }
    case kObjectiveNone: {
      *tot_weight = 0;
      *tot_objf = 0;
      break;
    }
    default:
      KALDI_ERR << "Objective function type " << objective_type
                << " not handled.";
  }
}



} // namespace nnet3
} // namespace kaldi
