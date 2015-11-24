// nnet3/nnet-example-test.cc

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

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-compile.h"
#include "nnet3/nnet-analyze.h"
#include "nnet3/nnet-test-utils.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "base/kaldi-math.h"

namespace kaldi {
namespace nnet3 {



void UnitTestNnetExample() {
  for (int32 n = 0; n < 50; n++) {

    NnetExample eg;
    int32 num_supervised_frames = RandInt(1, 10),
                   left_context = RandInt(0, 5),
                  right_context = RandInt(0, 5),
                      input_dim = RandInt(1, 10),
                     output_dim = RandInt(5, 10),
                    ivector_dim = RandInt(-1, 2);
    GenerateSimpleNnetTrainingExample(num_supervised_frames, left_context,
                                      right_context, input_dim, output_dim,
                                      ivector_dim, &eg);
    bool binary = (RandInt(0, 1) == 0);
    std::ostringstream os;
    eg.Write(os, binary);
    NnetExample eg_copy;
    if (RandInt(0, 1) == 0)
      eg_copy = eg; 
    std::istringstream is(os.str());
    eg_copy.Read(is, binary);
    std::ostringstream os2;
    eg_copy.Write(os2, binary);
    if (binary) {
      KALDI_ASSERT(os.str() == os2.str());
    }
    KALDI_ASSERT(ExampleApproxEqual(eg, eg_copy, 0.1));
  }
}


void UnitTestNnetMergeExamples() {
  for (int32 n = 0; n < 50; n++) {
    int32 num_supervised_frames = RandInt(1, 10),
                   left_context = RandInt(0, 5),
                  right_context = RandInt(0, 5),
                      input_dim = RandInt(1, 10),
                     output_dim = RandInt(5, 10),
                    ivector_dim = RandInt(-1, 2);

    int32 num_egs = RandInt(1, 4);
    std::vector<NnetExample> egs_to_be_merged(num_egs);
    for (int32 i = 0; i < num_egs; i++) {
      NnetExample eg;
      // sometimes omit the ivector.  just tests things a bit more
      // thoroughly.
      GenerateSimpleNnetTrainingExample(num_supervised_frames, left_context,
                                        right_context, input_dim, output_dim,
                                        RandInt(0, 1) == 0 ? 0 : ivector_dim,
                                        &eg);
      KALDI_LOG << i << "'th example to be merged is: ";
      eg.Write(std::cerr, false);
      egs_to_be_merged[i].Swap(&eg);
    }
    NnetExample eg_merged;
    bool compress = (RandInt(0, 1) == 0);
    MergeExamples(egs_to_be_merged, compress, &eg_merged);
    KALDI_LOG << "Merged example is: ";
    eg_merged.Write(std::cerr, false);
  }
}


void UnitTestNnetSplitExampleBySplitChunkInExample() {
  for (int32 n = 0; n < 50; n++) {
    int32 num_supervised_frames_after_split = RandInt(1, 10),
                   num_examples_after_split = RandInt(1,10),
         num_supervised_frames_before_split = 
	 num_supervised_frames_after_split * num_examples_after_split,
                               left_context = RandInt(0, 5),
                              right_context = RandInt(0, 5),
                                  input_dim = RandInt(1, 10),
                                 output_dim = RandInt(5, 10),
                                ivector_dim = RandInt(-1, 2);
    NnetExample eg;
    GenerateSimpleNnetTrainingExample(num_supervised_frames_before_split,
		                      left_context, right_context,
				      input_dim, output_dim,
				      RandInt(0, 1) == 0 ? 0 : ivector_dim,
                                      &eg);
    KALDI_LOG << n << "'th example to be splitted is: ";
    eg.Write(std::cerr, false);
    int32 num_chunks = -1;
    std::vector<NnetExample> egs_splitted;
    eg.SplitChunk(num_supervised_frames_after_split,
		  left_context, right_context, &num_chunks, &egs_splitted);
    // test if num of splitted examples agree
    KALDI_ASSERT(egs_splitted.size() == num_examples_after_split);

    int32 num_input_frames = num_supervised_frames_after_split
	                       + left_context + right_context;
    KALDI_LOG << num_examples_after_split << " splitted examples are: ";
    for (int32 f = 0; f < static_cast<int32>(eg.io.size()); f++) {
      Matrix<BaseFloat> feat;
      eg.io[f].features.GetMatrix(&feat);
      for (int32 i = 0; i < static_cast<int32>(egs_splitted.size()); i++) {
	const std::vector<NnetIo> &io = egs_splitted[i].io;
	// test if the number of data and indexes in a splitted example agree
        KALDI_ASSERT(io[f].features.NumRows() == io[f].indexes.size());
	// test if io names unchanged after split
	KALDI_ASSERT(io[f].name == eg.io[f].name);

	int32 row_offset = -1, num_rows = -1;
	if (io[f].name == "input") {
	  row_offset = i * num_supervised_frames_after_split;
	  num_rows = num_input_frames;
	  // test if the start of indexes "t" is unchanged after split
	  KALDI_ASSERT(io[f].indexes.begin()->t == eg.io[f].indexes.begin()->t);
	} else if (io[f].name == "output") {
	  row_offset = i * num_supervised_frames_after_split;
	  num_rows = num_supervised_frames_after_split;
	  KALDI_ASSERT(io[f].indexes.begin()->t == eg.io[f].indexes.begin()->t);

	} else if (io[f].name == "ivector") {
	  row_offset = 0;
	  num_rows = 1;
	  // test if indexes are unchanged after split
	  KALDI_ASSERT(io[f].indexes == eg.io[f].indexes);
	}
	SubMatrix<BaseFloat> feat_sub = feat.RowRange(row_offset, num_rows);
	Matrix<BaseFloat> feat_splitted;
	io[f].features.GetMatrix(&feat_splitted);
	// test if the data matrices are correctly splitted
	KALDI_ASSERT(ApproxEqual(feat_sub, feat_splitted,
				 static_cast<BaseFloat>(0.001)));

	if (f == 0) {
	  KALDI_LOG << i << "'th:";
	  egs_splitted[i].Write(std::cerr, false);
	  KALDI_LOG << "";
	}
      }
    } 
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  UnitTestNnetExample();
  UnitTestNnetMergeExamples();
  UnitTestNnetSplitExampleBySplitChunkInExample();

  KALDI_LOG << "Nnet-example tests succeeded.";

  return 0;
}

