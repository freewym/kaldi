// nnet3bin/nnet3-copy-egs-backstitch.cc

// Copyright 2012-2017  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar
//                2017  Yiming Wang

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

#include <queue>
#include <utility>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples for backstitch neural network training, possibly changing "
        "the binary mode.\n"
        "\n"
        "Usage:  nnet3-copy-egs-backstitch [options] <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "e.g.\n"
        "nnet3-copy-egs-backstitch ark:train.egs ark,t:text.egs\n"
        "See also: nnet3-subset-egs, nnet3-get-egs, nnet3-merge-egs, nnet3-shuffle-egs\n";

    BaseFloat backstitch_alpha = 0.0, backstitch_period = 1,
              backstitch_delay = 1000;

    ParseOptions po(usage);
    po.Register("backstitch-alpha", &backstitch_alpha, "backstitch scale.");
    po.Register("backstitch-period", &backstitch_period, "backstitch period.");
    po.Register("backstitch-delay", &backstitch_delay, "backstitch delay");


    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);
    SequentialNnetExampleReader example_reader(examples_rspecifier);

    std::string examples_wspecifier = po.GetArg(2);
    NnetExampleWriter example_writer(examples_wspecifier);

    int64 num_read = 0, num_written = 0;
    std::queue<std::pair<int64, std::pair<std::string, NnetExample> > > queue_egs;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      // To avoid outputting minibatches where the total negative weight is more
      // than the total positive weight.
      int32 current_backstitch_period = backstitch_period;
      if (num_read < backstitch_delay && current_backstitch_period < 2) {
        current_backstitch_period = 2;
      }
      std::string key = example_reader.Key();
      const NnetExample &eg = example_reader.Value();
      if (backstitch_alpha == 0.0) {
        example_writer.Write(key, eg);
        num_written++;
      } else {
        KALDI_ASSERT(backstitch_alpha > 0.0);
        // outputs a "negative" example with its output scaled by alpha every
        // current_backstitch_period examples, and push the unscaled version
        // into the queue
        if (num_read % current_backstitch_period == 0) {
          queue_egs.push(std::make_pair(num_read + backstitch_delay,
              std::make_pair(key, eg)));
          NnetExample eg_modified(eg);
          for (int32 i = 0; i < eg_modified.io.size(); i++) {
            if (eg_modified.io[i].name == "output") {
              KALDI_ASSERT(eg_modified.io[i].features.Type() == kSparseMatrix);
              SparseMatrix<BaseFloat> smat;
              eg_modified.io[i].features.SwapSparseMatrix(&smat);
              smat.Scale(-backstitch_alpha);
              eg_modified.io[i].features.SwapSparseMatrix(&smat);
              break;
            }
          }
          example_writer.Write(key + "_backstitch", eg_modified);
          num_written++;
        } else { // output normal examples
          example_writer.Write(key, eg);
          num_written++;
        }
        // outputs the "positive" example with its output scaled by (1+alpha)
        // after processing backstitch_delay more examples from the negative
        // example.
        if (queue_egs.front().first == num_read) {
          std::string &key_backstitch = queue_egs.front().second.first;
          NnetExample &eg_modified = queue_egs.front().second.second;
          for (int32 i = 0; i < eg_modified.io.size(); i++) {
            if (eg_modified.io[i].name == "output") {
              KALDI_ASSERT(eg_modified.io[i].features.Type() == kSparseMatrix);
              SparseMatrix<BaseFloat> smat;
              eg_modified.io[i].features.SwapSparseMatrix(&smat);
              smat.Scale(1.0 + backstitch_alpha);
              eg_modified.io[i].features.SwapSparseMatrix(&smat);
              break;
            }
          }
          example_writer.Write(key_backstitch, eg_modified);
          queue_egs.pop();
          num_written++;
        }
      }
    }

    while (!queue_egs.empty()) {
      std::string &key_backstitch = queue_egs.front().second.first;
      NnetExample &eg_modified = queue_egs.front().second.second;
      for (int32 i = 0; i < eg_modified.io.size(); i++) {
        if (eg_modified.io[i].name == "output") {
          KALDI_ASSERT(eg_modified.io[i].features.Type() == kSparseMatrix);
          SparseMatrix<BaseFloat> smat;
          eg_modified.io[i].features.SwapSparseMatrix(&smat);
          smat.Scale(1.0 + backstitch_alpha);
          eg_modified.io[i].features.SwapSparseMatrix(&smat);
          break;
        }
      }
      example_writer.Write(key_backstitch, eg_modified);
      queue_egs.pop();
      num_written++;
    }

    KALDI_LOG << "Read " << num_read << " neural-network training examples, wrote "
              << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
