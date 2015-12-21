// nnet3/nnet-utils-test.cc

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
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {


void UnitTestNnetContext() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationOptions gen_config;
    
    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    std::istringstream is(configs[0]);
    nnet.ReadConfig(is);

    // this test doesn't really test anything except that it runs;
    // we manually inspect the output.
    int32 left_context, right_context;
    ComputeSimpleNnetContext(nnet, &left_context, &right_context);
    KALDI_LOG << "Left,right-context= " << left_context << ","
              << right_context << " for config: " << configs[0];
  }
}

void UnitTestRecurrentNodeNamesAndOffsets() {
  struct NnetGenerationOptions gen_config;
  std::vector<std::string> configs;
  GenerateConfigSequenceStatePreservingLstm(gen_config, &configs);
  Nnet nnet;
  std::istringstream is(configs[0]);
  nnet.ReadConfig(is);

  const std::string arr[] = {"r_t", "c1_t", "c2_t"};
  std::vector<std::string> recurrent_node_names_truth(arr,
		  arr + sizeof(arr) / sizeof(arr[0]));
  
  std::vector<std::string> recurrent_output_names;
  std::vector<std::string> recurrent_node_names;
  std::vector<int32> recurrent_offsets;
  GetRecurrentOutputNodeNames(nnet, &recurrent_output_names,
		              &recurrent_node_names);
  GetRecurrentNodeOffsets(nnet, recurrent_node_names, &recurrent_offsets);
  for (int32 i = 0; i < recurrent_offsets.size(); i++) {
    KALDI_LOG << recurrent_node_names[i] << " offset=" << recurrent_offsets[i];
    std::vector<std::string>::iterator iter;
    iter = find(recurrent_node_names_truth.begin(),
                recurrent_node_names_truth.end(), recurrent_node_names[i]);
    KALDI_ASSERT(iter != recurrent_node_names_truth.end());
    KALDI_ASSERT(recurrent_offsets[i] == -1);
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  SetVerboseLevel(2);

  UnitTestNnetContext();
  UnitTestRecurrentNodeNamesAndOffsets();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}
