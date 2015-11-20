// nnet3/nnet-example.cc

// Copyright 2012-2015    Johns Hopkins University (author: Daniel Povey)
//                2014    Vimal Manohar

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

#include "nnet3/nnet-example.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet3 {

void NnetIo::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetIo>");
  WriteToken(os, binary, name);
  WriteIndexVector(os, binary, indexes);
  features.Write(os, binary);
  WriteToken(os, binary, "</NnetIo>");
  KALDI_ASSERT(static_cast<size_t>(features.NumRows()) == indexes.size());
}

void NnetIo::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetIo>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  features.Read(is, binary);
  ExpectToken(is, binary, "</NnetIo>");    
}

NnetIo::NnetIo(const std::string &name,
               int32 t_begin, const MatrixBase<BaseFloat> &feats):
    name(name), features(feats) {
  int32 num_rows = feats.NumRows();
  KALDI_ASSERT(num_rows > 0);
  indexes.resize(num_rows);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_rows; i++)
    indexes[i].t = t_begin + i;
}

NnetIo::NnetIo(const std::string &name,
               int32 dim,
               int32 t_begin,
               const Posterior &labels):
    name(name) {
  int32 num_rows = labels.size();
  KALDI_ASSERT(num_rows > 0);
  SparseMatrix<BaseFloat> sparse_feats(dim, labels);
  features = sparse_feats;
  indexes.resize(num_rows);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_rows; i++)
    indexes[i].t = t_begin + i;
}

int32 NnetIo::NumFramesPerChunk() const {
  std::vector<Index>::const_iterator iter = indexes.begin(),
                                      end = indexes.end();
  unordered_set<int32> frame_indexes;
  for (; iter != end; ++iter)
    if (iter->n == 0)
      frame_indexes.insert(iter->t);
  return static_cast<int32>(frame_indexes.size());
}

int32 NnetIo::NumChunks() const {
  std::vector<Index>::const_iterator iter = indexes.begin(),
	                              end = indexes.end();
  int32 n = 0;
  for (; iter != end; ++iter)
    n = std::max(iter->n, n);
  return n + 1;
}

void NnetExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<Nnet3Eg>");
  WriteToken(os, binary, "<NumIo>");
  int32 size = io.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    io[i].Write(os, binary);
  WriteToken(os, binary, "</Nnet3Eg>");
}

void NnetExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3Eg>");
  ExpectToken(is, binary, "<NumIo>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size < 0 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  io.resize(size);
  for (int32 i = 0; i < size; i++)
    io[i].Read(is, binary);
  ExpectToken(is, binary, "</Nnet3Eg>");
}

void NnetExample::SplitChunk(int32 new_chunk_size,
		             int32 left_context,
	                     int32 right_context,
		             int32 *ptr_num_chunks,
			     std::vector<NnetExample> *splitted) const {
  KALDI_ASSERT(new_chunk_size > 0);
  int32 old_chunk_size = -1, num_chunks = -1, num_input_frames_per_chunk = -1,
	num_minibatches = -1, extra_left_frames = -1, extra_right_frames = -1;

  for (int32 f = 0; f < static_cast<int32>(io.size()); f++)
    if (io[f].name == "output") {
      // compute the original chunk size, num of chunks in minibatch
      // and num of minibatches after splitting
      old_chunk_size = io[f].NumFramesPerChunk();
      num_chunks = io[f].NumChunks();
      *ptr_num_chunks = num_chunks;
      KALDI_ASSERT(old_chunk_size % new_chunk_size == 0);
      num_minibatches = old_chunk_size / new_chunk_size;
    }
  for (int32 f = 0; f < static_cast<int32>(io.size()); f++) 
    if (io[f].name == "input") {
      // compute num of input frames per chunk
      num_input_frames_per_chunk = io[f].NumFramesPerChunk();
      // compute extra left frames and extra right frames
      extra_left_frames = - io[f].indexes[0].t - left_context;
      extra_right_frames = num_input_frames_per_chunk - old_chunk_size
		           - left_context - right_context
                           - extra_left_frames;
      KALDI_LOG << "extra_left=" << extra_left_frames << " extra_right=" << extra_right_frames; //debug
    }

  KALDI_LOG << "num_minibatches=" << num_minibatches << " input_frames_per_chunk=" << num_input_frames_per_chunk << " num_chunks=" << num_chunks; //debug
  splitted->clear();
  splitted->resize(num_minibatches);
  // do splitting
  std::vector<NnetExample>::iterator iter = splitted->begin(),
	                             end = splitted->end();
  for (int32 i = 0; iter != end; iter++, i++) {
    NnetExample &eg = *iter;
    eg.io.resize(io.size());
    int32 num_feats = static_cast<int32>(io.size());
    std::vector<std::vector<GeneralMatrix const*> > output_lists(num_feats);
    std::vector<std::vector<GeneralMatrix> > output_matrices(num_feats);
    for (int32 f = 0; f < num_feats; f++) {
      eg.io[f].name = io[f].name;
      output_matrices[f].resize(num_chunks);
      if (io[f].name == "output" || io[f].NumFramesPerChunk() 
	  == old_chunk_size) { // output or other NnetIo that has the same size
        eg.io[f].indexes.resize(num_chunks * new_chunk_size);
	for (int32 n = 0; n < num_chunks; n++) {
	  int32 src_begin_pos = n * old_chunk_size + i * new_chunk_size,
	        src_end_pos = src_begin_pos + new_chunk_size,
                dst_begin_pos = n * new_chunk_size;
	  // copy indexes
	  std::copy(io[f].indexes.begin() + src_begin_pos,
		    io[f].indexes.begin() + src_end_pos,
		    eg.io[f].indexes.begin() + dst_begin_pos);
	  // modify indexes "t"
	  for (int32 t = 0; t < new_chunk_size; t++)
            eg.io[f].indexes[dst_begin_pos + t].t = t;
	  // copy corresponding features
	  std::vector<bool> keep_rows(io[f].features.NumRows(), false);
	  for (int32 j = src_begin_pos; j < src_end_pos; j++)
	    keep_rows[j] = true;
	  FilterGeneralMatrixRows(io[f].features, keep_rows,
			          &(output_matrices[f][n]));
	  output_lists[f].push_back(&(output_matrices[f][n]));
	}
      } else if (io[f].name == "input") { // input
        // new chunks in the first/last minibatches should also include
        // all extra frames before/after left_context/right_context from 
        // the old chunks (e.g. chunk_left_context/chunk_right_context)
	eg.io[f].indexes.resize(num_chunks * (new_chunk_size
		+ left_context + right_context
		+ (i == 0 ? extra_left_frames : 0)
		+ (i == num_minibatches -1 ? extra_right_frames : 0)));
	for (int32 n = 0; n < num_chunks; n++) {
	  int32 src_begin_pos = n * num_input_frames_per_chunk
			+ i * new_chunk_size
			+ (i == 0 ? 0 : extra_left_frames),
	        src_end_pos = src_begin_pos + new_chunk_size
			+ left_context + right_context
			+ (i == 0 ? extra_left_frames : 0)
			+ (i == num_minibatches - 1 ? extra_right_frames : 0),
	        dst_begin_pos = n * (new_chunk_size 
			+ left_context + right_context
		        + (i == 0 ? extra_left_frames : 0)
			+ (i == num_minibatches - 1 ? extra_right_frames : 0));
	  //copy indexes
          std::copy(io[f].indexes.begin() + src_begin_pos,
	            io[f].indexes.begin() + src_end_pos,
		    eg.io[f].indexes.begin() + dst_begin_pos);
	  // modify indexes "t"
	  int32 t = -left_context - (i == 0 ? extra_left_frames : 0);
	  for (int32 j = 0; j < src_end_pos - src_begin_pos; j++, t++)
	    eg.io[f].indexes[dst_begin_pos + j].t = t;
	  // copy corresponding features
	  std::vector<bool> keep_rows(io[f].features.NumRows(), false);
	  for (int32 j = src_begin_pos; j < src_end_pos; j++)
	    keep_rows[j] = true;
          FilterGeneralMatrixRows(io[f].features, keep_rows,
			          &(output_matrices[f][n]));
	  output_lists[f].push_back(&(output_matrices[f][n]));
	}
      } else if (io[f].NumFramesPerChunk() == 1) { // e.g. ivector
        eg.io[f].indexes.resize(num_chunks);
	for (int32 n = 0; n < num_chunks; n++) {
	  int32 src_begin_pos = n, src_end_pos = src_begin_pos + 1,
		dst_begin_pos = n;
	  // copy indexes
	  std::copy(io[f].indexes.begin() + src_begin_pos,
		    io[f].indexes.begin() + src_end_pos,
		    eg.io[f].indexes.begin() + dst_begin_pos);
	  // copy corresponding features
	  std::vector<bool> keep_rows(io[f].features.NumRows(), false);
	  for (int32 j = src_begin_pos; j < src_end_pos; j++)
	    keep_rows[j] = true;
          FilterGeneralMatrixRows(io[f].features, keep_rows,
			          &(output_matrices[f][n]));
	  output_lists[f].push_back(&(output_matrices[f][n]));
	}
      }
      AppendGeneralMatrixRows(output_lists[f], &(eg.io[f].features));
    }
  }
}

void NnetExample::Compress() {
  std::vector<NnetIo>::iterator iter = io.begin(), end = io.end();
  // calling features.Compress() will do nothing if they are sparse or already
  // compressed.
  for (; iter != end; ++iter)
    iter->features.Compress();
}

} // namespace nnet3
} // namespace kaldi
