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
        "Add recurrent IOs to examples according to the input/output changes of nnet\n"
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

    // compute model left/right context
    int32 left_context = 0, right_context = 0;
    ComputeSimpleNnetContext(nnet, &left_context, &right_context);

    // extract recurrent output names and their offsets from nnet 
    std::vector<std::string> recurrent_output_names;
    std::vector<std::string> recurrent_node_names; 
    std::vector<int32> recurrent_offsets;
    GetRecurrentOutputNodeNames(nnet, &recurrent_output_names,
                                &recurrent_node_names);
    GetRecurrentNodeOffsets(nnet, recurrent_node_names, &recurrent_offsets);
    for (int32 i = 0; i < recurrent_offsets.size(); i++)
      KALDI_LOG << "recurrent node: " << recurrent_node_names[i] 
                << ", offset: " << recurrent_offsets[i];

    int64 num_read = 0, num_written = 0;

    while (!example_reader.Done()) {
      const std::string &cur_key = example_reader.Key();
      NnetExample cur_eg(example_reader.Value());
      example_reader.Next();
      num_read++;

      // compute the chunk size,  num of chunks of each minibatch and 
      // the begining output "t" index of each chunk in that minibatch
      int32 chunk_size = 0, num_chunks = 0;
      std::vector<int32> output_t_begin;
      for (int32 f = 0; f < cur_eg.io.size(); f++)
        if (cur_eg.io[f].name == "output") {
          chunk_size = NumFramesPerChunk(cur_eg.io[f]);
          num_chunks = NumChunks(cur_eg.io[f]);
          output_t_begin.reserve(num_chunks);
          for (int32 n = 0; n < num_chunks; n++)
            output_t_begin.push_back((cur_eg.io[f].indexes.begin() +
                                     n * chunk_size)->t);
          break;
        }
      int32 extra_left_frames = 0, extra_right_frames = 0;
      for (int32 f = 0; f < cur_eg.io.size(); f++) 
        if (cur_eg.io[f].name == "input") {
          // compute num of input frames per chunk
          int32 num_input_frames_per_chunk = NumFramesPerChunk(cur_eg.io[f]);
          // compute extra left frames and extra right frames
          extra_left_frames = output_t_begin[0] - 
                              cur_eg.io[f].indexes.begin()->t - left_context;
          extra_right_frames = num_input_frames_per_chunk - chunk_size -
                               left_context - right_context - extra_left_frames;
          break;
      }

      for (int32 i = 0; i < recurrent_output_names.size(); i++) {
        const std::string &node_name = recurrent_output_names[i];
        const int32 offset = recurrent_offsets[i];
        KALDI_ASSERT(offset != 0);

        // create zero matrix for input
        Matrix<BaseFloat> zero_matrix_as_input(num_chunks * abs(offset),
                                               nnet.OutputDim(node_name));
        // Add recurrent inputs to each NnetExample's io 
        cur_eg.io.push_back(NnetIo(node_name + "_STATE_PREVIOUS_MINIBATCH", 0,
                            zero_matrix_as_input));
        // Correct the indexes: so that "n" ranges [0, feats.NumRows() - 1] and
        // "t" ranges [output_t_begin[n] - extra_left_frames + offset,
        // output_t_begin[n] - extra_left_frames - 1] (if offset < 0) or
        // [output_t_begin[n] + extra_right_frames + chunk_size,
        // output_t_begin[n] + extra_right_frames + chunk_size + offset - 1]
        // (if offset > 0). 
        // The assumption is that the n-th chunk's output frames "t" indexes
        // ranges [output_t_begin[n], output_t_begin[n] + num_chunks - 1]
        std::vector<Index> &indexes_input = cur_eg.io.back().indexes;
        for (int32 n = 0; n < num_chunks; n++)
          for (int32 t = 0; t < abs(offset); t++) {
            int32 j = n * abs(offset) + t;
            indexes_input[j].n = n;
            if (offset < 0)
              indexes_input[j].t = output_t_begin[n] - extra_left_frames +
                                   offset + t;
            else
              indexes_input[j].t = output_t_begin[n] + extra_right_frames +
                                   chunk_size + t;
          }

        // create zero matrix for output
        Matrix<BaseFloat> zero_matrix_as_output(num_chunks * chunk_size,
                                                nnet.OutputDim(node_name));
        // Add recurrent outputs to each NnetExample's io.
        // Actually the contents of output matrix is irrelevant; we don't need
        // it as supervision; we only need its NunRows info for NnetIo::indexes.
        // So we just use zero matrix.
        cur_eg.io.push_back(NnetIo(node_name, 0, zero_matrix_as_output));
        // correct the indexes.
        std::vector<Index> &indexes_output = cur_eg.io.back().indexes;
        for (int32 n = 0; n < num_chunks; n++)
          for (int32 t = 0; t < chunk_size; t++) {
            int32 j = n * chunk_size + t;
            indexes_output[j].n = n;
            indexes_output[j].t = output_t_begin[n] + t;
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
