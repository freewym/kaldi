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
        "Add additional IOs to examples according to the input/output changes of nnet\n"
        "for state preserving LSTM training.\n"
        "\n"
        "Usage:  nnet3-add-recurrent-io-to-egs [options] <raw-model-in> <egs-in> <egs-out>\n"
        "\n"
        "An example:\n"
        "nnet3-add-recurrent-io-to-egs 1.raw 1.egs ark:- \n";

    bool compress = true;
        
    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    
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

    // extract recurrent output names from nnet
    std::vector<std::string> recurrent_output_names;
    GetRecurrentOutputNodeNames(nnet, &recurrent_output_names);

    int64 num_read = 0, num_written = 0;

    while (!example_reader.Done()) {
      const std::string &cur_key = example_reader.Key();
      NnetExample cur_eg(example_reader.Value());
      example_reader.Next();
      num_read++;

      // compute the chunk size and num of chunks of each minibatch
      int32 chunk_size = -1, num_chunks = -1;
      for (int32 f = 0; f < cur_eg.io.size(); f++)
        if (cur_eg.io[f].name == "output") {
          chunk_size = NumFramesPerChunk(cur_eg.io[f]);
          num_chunks = NumChunks(cur_eg.io[f]);
          break;
        }

      for (int32 i = 0; i < recurrent_output_names.size(); i++) {
        const std::string &node_name = recurrent_output_names[i];

        // create zero matrix for input
        Matrix<BaseFloat> zero_matrix_as_input(num_chunks,
                                               nnet.OutputDim(node_name));
        // Add to NnetIo the recurrent connections from
        // the previous minibatch as additional inputs
        cur_eg.io.push_back(NnetIo(node_name + "_STATE_PREVIOUS_MINIBATCH", 0,
                            zero_matrix_as_input));
        // Correct the indexes: swap indexes "n" and "t" so that 
        // n ranges from 0 to feats.NumRows() - 1 and t is always 0
        std::vector<Index> &indexes_input = cur_eg.io.back().indexes;
        for (int32 j = 0; j < indexes_input.size(); j++)
          std::swap(indexes_input[j].n, indexes_input[j].t);

        // create zero matrix for output
        Matrix<BaseFloat> zero_matrix_as_output(num_chunks * chunk_size,
                                                nnet.OutputDim(node_name));
        // Add to NnetIo the recurrent connections in the current minibatch 
        // as additional outputs. Actually the contents of output matrix is
        // irrelevant; we don't need it as supervision; we only need its 
        // NunRows info for NnetIo::indexes. So we just use zero matrix.
        cur_eg.io.push_back(NnetIo(node_name, 0, zero_matrix_as_output));
        // correct the indexes.
        std::vector<Index> &indexes_output = cur_eg.io.back().indexes;
        for (int32 n = 0, j = 0; n < num_chunks; n++)
          for (int32 t = 0; t < chunk_size; t++, j++) {
            indexes_output[j].n = n;
            indexes_output[j].t = t;
          }
      }

      if (compress)
        cur_eg.Compress();
      example_writer.Write(cur_key, cur_eg);
      num_written++;
    }
    KALDI_LOG << "Processed " << num_read << " egs to " << num_written << "."; 
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
