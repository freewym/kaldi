// nnet3bin/nnet3-compute-from-egs.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-utils.h"
#include "transform/lda-estimate.h"


namespace kaldi {
namespace nnet3 {

class NnetComputerFromEg {
 public:
  NnetComputerFromEg(const Nnet &nnet):
      nnet_(nnet), compiler_(nnet) { }

  // Compute the output (which will have the same number of rows as the number
  // of Indexes in the output of the eg), and put it in "output".
  void Compute(const NnetExample &eg, Matrix<BaseFloat> *output) {
    // make a copy of eg so that we can keep eg const
    NnetExample eg_modified(eg);

    int32 num_chunks = -1, chunk_size = -1;
    for (int32 f = 0; f < static_cast<int32>(eg.io.size()); f++)
      if (eg.io[f].name == "output") {
        num_chunks = eg.io[f].NumChunks();
        chunk_size = eg.io[f].NumFramesPerChunk();
      }
    KALDI_LOG << "num_chunks=" << num_chunks << " chunk_size=" << chunk_size;//debug

    std::vector<std::string> recurrent_output_names;
    std::vector<std::string> recurrent_node_names;
    GetRecurrentOutputNodeNames(nnet_, &recurrent_output_names,
		                &recurrent_node_names);

    for (int32 i = 0; i < static_cast<int32>(recurrent_output_names.size());
         i++) {
      std::string &node_name = recurrent_output_names[i];
      KALDI_LOG <<"node_name=" <<node_name;//debug

      // Add to NnetIo the recurrent connections as additional inputs
      Matrix<BaseFloat> zero_matrix_as_input = Matrix<BaseFloat>(
		      num_chunks, nnet_.OutputDim(node_name));
      eg_modified.io.push_back(NnetIo(recurrent_output_names[i]
	      + "_STATE_PREVIOUS_MINIBATCH", 0, zero_matrix_as_input));
      // Correct the indexes: swap indexes "n" and "t" so that 
      // n ranges from 0 to feats.NumRows() - 1 and t is always 0
      std::vector<Index> &indexes_input = eg_modified.io.back().indexes;
      for (int32 j = 0; j < static_cast<int32>(indexes_input.size()); j++)
        std::swap(indexes_input[j].n, indexes_input[j].t);

      // Add to NnetIo the recurrent connections in the current minibatch 
      // as additional outputs. Actually the contents of output matrix is
      // irrelevant; we don't need it as supervision; we only need its 
      // NunRows info for NnetIo::indexes. So we just use zero matrix.
      Matrix<BaseFloat> zero_matrix_as_output = Matrix<BaseFloat>(
		      num_chunks * chunk_size, nnet_.OutputDim(node_name));
      eg_modified.io.push_back(NnetIo(recurrent_output_names[i], 0,
		  zero_matrix_as_output));
      // correct the indexes.
      std::vector<Index> &indexes_output = eg_modified.io.back().indexes;
      for (int32 n = 0, j = 0; n < num_chunks; n++)
        for (int32 t = 0; t < chunk_size; t++, j++) {
	  indexes_output[j].n = n;
	  indexes_output[j].t = t;
        } 
    }

    ComputationRequest request;
    bool need_backprop = false, store_stats = false;
    GetComputationRequest(nnet_, eg_modified, need_backprop, store_stats,
		          &request);
    const NnetComputation &computation = *(compiler_.Compile(request));
    NnetComputeOptions options;
    if (GetVerboseLevel() >= 3)
      options.debug = true;
    NnetComputer computer(options, computation, nnet_, NULL);
    computer.AcceptInputs(nnet_, eg_modified.io);
    computer.Forward();
    const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
    output->Resize(nnet_output.NumRows(), nnet_output.NumCols());
    nnet_output.CopyToMat(output);
  }
 private:
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;
  
};

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Read input nnet training examples, and compute the output for each one.\n"
        "If --apply-exp=true, apply the Exp() function to the output before writing\n"
        "it out.\n"
        "\n"
        "Usage:  nnet3-compute-from-egs [options] <raw-nnet-in> <training-examples-in> <matrices-out>\n"
        "e.g.:\n"
        "nnet3-compute-from-egs --apply-exp=true 0.raw ark:1.egs ark:- | matrix-sum-rows ark:- ... \n"
        "See also: nnet3-compute\n";
    
    bool binary_write = true,
        apply_exp = false;
    std::string use_gpu = "yes";

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("apply-exp", &apply_exp, "If true, apply exp function to "
                "output");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        matrix_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetComputerFromEg computer(nnet);

    int64 num_egs = 0;
    
    SequentialNnetExampleReader example_reader(examples_rspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    
    for (; !example_reader.Done(); example_reader.Next(), num_egs++) {
      Matrix<BaseFloat> output;
      computer.Compute(example_reader.Value(), &output);
      KALDI_ASSERT(output.NumRows() != 0);
      if (apply_exp)
        output.ApplyExp();
      matrix_writer.Write(example_reader.Key(), output);
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    KALDI_LOG << "Processed " << num_egs << " examples.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


