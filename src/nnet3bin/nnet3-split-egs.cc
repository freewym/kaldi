#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Split large chunks in examples into smaller chunks of equal size\n"
        "by generating more examples containing these smaller chunks,\n"
        "while still preserving the number of left/right context for\n"
        "each new chunk.\n"
        "\n"
        "Usage:  nnet3-split-egs [options] <raw-model-in> <egs-in> <egs-out>\n"
        "\n"
        "An example:\n"
        "nnet3-split-egs --chunk-size-after-split=20 1.raw \\\n"
        "'ark:nnet3-merge-egs 1.egs ark:-|' ark:- \n ";

    int32 chunk_size_after_split = 0;
        
    ParseOptions po(usage);
    po.Register("chunk-size-after-split", &chunk_size_after_split,
		"used for enabling state preserving LSTM. default 0 indicates "
		"no state preserving");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string nnet_rxfilename = po.GetArg(1),
         examples_rspecifier = po.GetArg(2),
         examples_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    int32 left_context = 0, right_context = 0;
    KALDI_ASSERT(!example_reader.Done());
    ComputeSimpleNnetContext(nnet, &left_context, &right_context);
    KALDI_LOG << "model left_context=" << left_context
              << ", model right_context=" << right_context;

    std::vector<NnetExample> splitted_examples;
    
    int64 num_read = 0, num_written = 0;

    while (!example_reader.Done()) {
      const std::string &cur_key = example_reader.Key();
      const NnetExample &cur_eg = example_reader.Value();
      SplitChunk(chunk_size_after_split, left_context, right_context, cur_eg,
                 &splitted_examples);
      example_reader.Next();
      num_read++;

      for (size_t i = 0; i < splitted_examples.size(); i++) {
        std::ostringstream ostr;
        ostr << cur_key << "-split-" << i;
        std::string output_key = ostr.str();
        example_writer.Write(output_key, splitted_examples[i]);
        num_written++;
      }
      splitted_examples.clear();
    }
    KALDI_LOG << "Splitted " << num_read << "egs to " << num_written << "."; 
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
