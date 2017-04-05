// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#include <DDImage/Iop.h>
#include <DDImage/Row.h>
#include <DDImage/Knobs.h>
#include <DDImage/ImagePlane.h>
#include <DDImage/NukeWrapper.h>
#include <DDImage/Tile.h>
#include <DDImage/Thread.h>

#include <boost/bind.hpp>
#include <cuda_runtime.h>
#include <math.h>

#include <vector>
#include <algorithm>

#include "include/threading.h"
#include "include/cuda_helper.h"
#include "include/nuke_helper.h"
#include "include/median.h"

#define ThisClass AFXMedian

namespace nuke = DD::Image;

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXMedian";
static const char* HELP = "Smart Median Filter";

extern "C" void MedianCuda(cudaTextureObject_t tex, float* row_ptr, size_t row_pitch, int med_size_i_, int med_size_o, int med_n_i,
                           int med_n_o, float m_lerp, float m_i_lerp, float sharpness, afx::Bounds img_bnds, afx::Bounds row_bnds, cudaStream_t stream);

class ThisClass : public nuke::Iop {
 private:
  // members to store knob values
  float k_size_, k_sharpness_;
  // members to store processed knob values
  int med_size_i_, med_size_o_, med_n_i_, med_n_o_;
  float med_size_f_, m_lerp_, m_i_lerp_, sharpness_;

  boost::mutex mutex_;
  bool first_time_GPU_;
  bool first_time_CPU_;
  nuke::Lock lock_;

  afx::Bounds req_bnds_, format_bnds_, format_f_bnds_;
  float proxy_scale_;

  afx::CudaProcess cuda_process_;
  afx::CudaImageArray row_imgs_;
  afx::CudaImageArray in_imgs_;
  afx::CudaStreamArray streams_;

  void ProcessCUDA(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);
  void ProcessCPU(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);

 public:
  explicit ThisClass(Node* node);
  void knobs(nuke::Knob_Callback);
  const char* Class() const;
  const char* node_help() const;
  static const Iop::Description d;

  void _validate(bool);
  void _request(int x, int y, int r, int t, nuke::ChannelMask channels, int count);
  void _open();
  void _close();
  void engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);
};
ThisClass::ThisClass(Node* node) : Iop(node) {
  // Set inputs
  inputs(1);

  first_time_GPU_ = true;
  first_time_CPU_ = true;

  try {
    cuda_process_.MakeReady();
  } catch (cudaError err) {} // TODO printf

  // initialize knobs
  k_size_ = 1.0f;
  k_sharpness_ = 0.5f;
}
void ThisClass::knobs(nuke::Knob_Callback f) {
  nuke::Float_knob(f, &k_size_, "size", "Size");
  nuke::Tooltip(f, "Median Size");
  nuke::SetRange(f, 0.0, 10.0);

  nuke::Float_knob(f, &k_sharpness_, "sharpness", "Sharpness");
  nuke::Tooltip(f, "Sharpen high contrast regions");
  nuke::SetRange(f, 0.0, 1.0);
}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static nuke::Iop* build(Node* node) { return (new nuke::NukeWrapper(new ThisClass(node)))->channelsRGBoptionalAlpha(); }
const nuke::Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Smart Median", build);
void ThisClass::_validate(bool) {
  // Copy source info
  copy_info(0);

  format_bnds_ = afx::BoxToBounds(input(0)->format());
  format_f_bnds_ = afx::BoxToBounds(input(0)->full_size_format());
  proxy_scale_ = static_cast<float>(format_bnds_.GetWidth()) / static_cast<float>(format_f_bnds_.GetWidth());

  m_lerp_ = fmod(afx::math::Clamp<float>(proxy_scale_ * k_size_, 0.0f, 25.0f), 1.0f);
  m_i_lerp_ = (1.0f - m_lerp_);
  med_size_f_ = afx::math::Clamp<float>(proxy_scale_ * k_size_, 0.0f, 25.0f);
  med_size_i_ = static_cast<int>(med_size_f_);
  med_size_o_ = static_cast<int>(std::ceil(med_size_f_));
  med_n_i_ = (med_size_i_ * 2 + 1) * (med_size_i_ * 2 + 1);
  med_n_o_ = (med_size_o_ * 2 + 1) * (med_size_o_ * 2 + 1);
  sharpness_ = afx::math::Clamp<float>(k_sharpness_, 0.0f, 3.0f);
}
void ThisClass::_request(int x, int y, int r, int t, nuke::ChannelMask channels, int count) {
  req_bnds_.SetBounds(x, y, r - 1, t - 1);
  input0().request(afx::BoundsToBox(req_bnds_.GetPadBounds(med_size_o_)), channels, count);
}
void ThisClass::_open() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  in_imgs_.Clear();
  row_imgs_.Clear();
  streams_.Clear();
}
void ThisClass::_close() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  in_imgs_.Clear();
  row_imgs_.Clear();
  streams_.Clear();
}
void ThisClass::engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  callCloseAfter(0);  // Call close to dispose of all cuda mem allocations after last engine call
  try {
    ProcessCUDA(y, x, r, channels, row);
  } catch (cudaError err) {
    ProcessCPU(y, x, r, channels, row);
  }
}
void ThisClass::ProcessCUDA(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);

  {
    nuke::Guard guard(lock_);
    if (first_time_GPU_) {
      cuda_process_.CheckReady();  // Throw error if device is not ready

      afx::Bounds req_pad_bnds = req_bnds_.GetPadBounds(med_size_o_);
      req_pad_bnds.Intersect(afx::InputBounds(input(0)));

      // Create plane of input channels for request bounds. Edge repication is handled by cuda texture
      nuke::ImagePlane in_plane(afx::BoundsToBox(req_pad_bnds), false, channels);  // Create plane "false" = non-packed.
      if (aborted()) { return; }  // Return if render aborted
      input0().fetchPlane(in_plane);  // Fetch plane
      afx::CudaStreamArray streams;
      foreach(z, in_plane.channels()) {  // For each channel in plane
        in_imgs_.AddImage(req_pad_bnds);
        in_imgs_.GetBackPtr()->AddAttribute("channel", z);  // Add channel attribute to image
        streams.Add();  // Add stream
        in_imgs_.GetBackPtr()->MemCpyToDevice(&in_plane.readable()[in_plane.chanNo(z) * in_plane.chanStride()], in_plane.rowStride() * sizeof(float),
                                              streams.GetBackPtr()->GetStream());  // Copy mem host plane to cuda on stream.
      }
      cuda_process_.Synchonize();  // Sync all streams
      in_imgs_.CreateTextures();  // Create texture objects for all cuda images
      first_time_GPU_ = false;
    }
  }  // End first time guard

  if (aborted()) { return; }

  afx::CudaImage cuda_row_img(row_bnds);
  afx::CudaStream cuda_row_stream;
  foreach(z, channels) {
    afx::CudaImage* in_ptr = in_imgs_.GetPtrByAttribute("channel", z);
    MedianCuda(in_ptr->GetTexture(), cuda_row_img.GetPtr(), cuda_row_img.GetPitch(), med_size_i_, med_size_o_, med_n_i_, med_n_o_, m_lerp_,
               m_i_lerp_, sharpness_, in_ptr->GetBounds(), row_bnds, cuda_row_stream.GetStream());
    cuda_row_stream.Synchonize();
    cuda_row_img.MemCpyFromDevice(row.writable(z) + row_bnds.x1(), cuda_row_img.GetWidth() * sizeof(float));
  }
}
void ThisClass::ProcessCPU(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);
  afx::Bounds tile_bnds = row_bnds.GetPadBounds(med_size_o_);

  // Must call get before initializing any pointers
  row.get(input0(), y, x, r, channels);

  foreach(z, channels) {
    // Get the input to compute median. Fill tile without multithreading.
    nuke::Tile tile(input0(), tile_bnds.x1(), tile_bnds.y1(), tile_bnds.x2() + 1, tile_bnds.y2() + 1, z, false);
    // The constructor may abort, you can't look at tile then:
    if (aborted()) { return; }
    if (!tile.valid()) {
      foreach(z, channels) {
        float* out_ptr = row.writable(z);
        memset(&out_ptr[tile.x()], 0, (tile.r() - tile.x()) * sizeof(float));
      }
      return;
    }
    int med_size_2 = med_size_o_ * 2;
    const float* in_ptr = row[z] + row_bnds.x1();
    float* out_ptr = row.writable(z) + row_bnds.x1();
    for (int row_x = row_bnds.x1(); row_x <= row_bnds.x2(); ++row_x) {
      float sum_i = 0.0f, sum_o = 0.0f, sum_sq_i = 0.0f, sum_sq_o = 0.0f;
      std::vector<float> v_i;
      std::vector<float> v_o;
      v_i.reserve(med_n_i_);
      v_o.reserve(med_n_o_);
      int y_i = 0;  // Counter to find border pixels of median window
      for (int y0 = (row_bnds.y1() - med_size_o_); y0 <= (row_bnds.y2() + med_size_o_); ++y0) {
        int x_i = 0;  // Counter to find border pixels of median window
        int y_clamp = tile.clampy(y0);
        for (int x0 = (row_x - med_size_o_); x0 <= (row_x + med_size_o_); ++x0) {
          // Tile is not accessible outside of requested reqion. Must clamp to bounds.
          float value = tile[z][y_clamp][tile.clampx(x0)];
          // If on outside ring of pixels
          if (y_i == 0 || y_i == med_size_2 || x_i == 0 || x_i == med_size_2) {
            // Outside sum and sumsq will be added to inside later. Saves cpu time.
            sum_o += value;
            sum_sq_o += value * value;
          } else {  // If on inside ring of pixels
            sum_i += value;
            sum_sq_i += value * value;
            v_i.push_back(value);
          }
          v_o.push_back(value);
          x_i++;
        }
        y_i++;
      }
      float median = 0.0f;
      float mean = 0.0f;
      float std_dev = 0.0f;
      if (m_lerp_ == 0) {  // Integer median size, use outside list for stats
        median = afx::MedianQuickSelect(v_o.data(), v_o.size());
        mean = sum_o / std::max(med_n_o_, 1);
        std_dev = sqrtf(fmaxf((sum_sq_o) / std::max(med_n_o_, 1) - mean * mean, 0.0f));
      } else {  // Lerp inner and outter stats
        float median_i = afx::MedianQuickSelect(v_i.data(), v_o.size());
        float median_o = afx::MedianQuickSelect(v_o.data(), v_o.size());
        // lerp between inner median and full median
        median = m_i_lerp_ * median_i + m_lerp_ * median_o;
        float mean_i = sum_i / std::max(med_n_i_, 1);
        float std_dev_i = sqrtf(fmaxf(sum_sq_i / std::max(med_n_i_, 1) - mean_i * mean_i, 0.0f));
        float mean_o = (sum_i + sum_o) / std::max(med_n_o_, 1);
        mean = m_i_lerp_ * mean_i + m_lerp_ * mean_o;
        float std_dev_o = sqrtf(fmaxf((sum_sq_i + sum_sq_o) / std::max(med_n_o_, 1) - mean_o * mean_o, 0.0f));
        std_dev = m_i_lerp_ * std_dev_i + m_lerp_ * std_dev_o;
      }
      // Multiply std_dev by sharpness_ parameter
      std_dev = afx::math::Clamp<float>(sharpness_ * 3.0f * std_dev, 0.0f, 1.0f);
      // Lerp median and input value by std_dev
      *out_ptr = (1.0f - std_dev) * median + std_dev * (*in_ptr);
      in_ptr++;
      out_ptr++;
    }
  }
}

