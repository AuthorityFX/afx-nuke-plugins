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
#include <DDImage/Thread.h>
#include <DDImage/NukeWrapper.h>

#include <boost/bind.hpp>
#include <cuda_runtime.h>
#include <math.h>

#include "threading.h"
#include "image.h"
#include "cuda_helper.h"
#include "mlaa.h"

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXMonteCarloNoiseMap";
static const char* HELP = "Monte Carlo Noise Map";

#define ThisClass AFXMonteCarloNoiseMap

using namespace DD::Image;

class ThisClass : public Iop {
private:

  boost::mutex mutex_;
  bool first_time_GPU_;
  bool first_time_CPU_;
  Lock lock_;

  afx::Bounds req_bnds_, format_bnds, format_f_bnds_;
  float proxy_scale_;

  afx::CudaProcess cuda_process_;
  afx::CudaImageArray row_imgs_;
  afx::CudaImageArray in_imgs_;
  afx::CudaStreamArray streams_;

  int maximum_inputs() const { return 4; }
  int minimum_inputs() const { return 4; }

  void ProcessCUDA(int y, int x, int r, ChannelMask channels, Row& row);
  void ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row);

public:
  ThisClass(Node* node);
  void knobs(Knob_Callback);
  const char* Class() const;
  const char* node_help() const;
  static const Iop::Description d;

  void _validate(bool);
  void _request(int x, int y, int r, int t, ChannelMask channels, int count);
  void _open();
  void _close();
  void engine(int y, int x, int r, ChannelMask channels, Row& row);
};
ThisClass::ThisClass(Node* node) : Iop(node) {

  first_time_GPU_ = true;
  first_time_CPU_ = true;

  try {
    cuda_process_.MakeReady();
  } catch (cudaError err) {}
}
void ThisClass::knobs(Knob_Callback f) {

}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static Iop* build(Node* node) { return new ThisClass(node); }
const Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX MC Noise Map", build);

void ThisClass::_validate(bool) {
  //Copy source info
  copy_info(0);
  req_bnds_.SetBounds(info_.x(), info_.y(), info_.r() - 1, info_.t() - 1);
  format_bnds.SetBounds(input(0)->format().x(), input(0)->format().y(), input(0)->format().r() - 1, input(0)->format().t() - 1);
  format_f_bnds_.SetBounds(input(0)->full_size_format().x(), input(0)->full_size_format().y(), input(0)->full_size_format().r() - 1,
                           input(0)->full_size_format().t() - 1);
  proxy_scale_ = (float)format_bnds.GetWidth() / (float)format_f_bnds_.GetWidth();

  // Will store noise map in alpha channel
  ChannelSet add_channels = channels();
  add_channels += Chan_Alpha;
  set_out_channels(add_channels);
  info_.turn_on(add_channels);
}
void ThisClass::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
  //Request source
  input0().request(x, y, r, t, channels, count);
}
void ThisClass::_open() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
}
void ThisClass::_close() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  in_imgs_.Clear();
  row_imgs_.Clear();
  streams_.Clear();
}
void ThisClass::engine(int y, int x, int r, ChannelMask channels, Row& row) {
  callCloseAfter(0);
  try {
    throw cudaErrorIllegalAddress;
    ProcessCUDA(y, x, r, channels, row);
  } catch (cudaError err) {
    ProcessCPU(y, x, r, channels, row);
  }
}
void ThisClass::ProcessCUDA(int y, int x, int r, ChannelMask channels, Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);
}
void ThisClass::ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);

  if (first_time_CPU_) {
    Guard guard(lock_);
    if (first_time_CPU_) {
      Box plane_req_box(info_.box());
      // TODO Make channel knob to specify source
      ImagePlane source_plane(plane_req_box, false, Mask_RGB); // Create plane "false" = non-packed.
      input(0)->fetchPlane(source_plane);



      first_time_CPU_ = false;
    }
  } // End first time guard

  // Remove alpha from channels and copy all other channels straight through to output row
  ChannelSet copy_mask = channels - Mask_Alpha;
  row.get(input0(), y, x, r, channels);
  row.pre_copy(row, copy_mask);
  row.copy(row, copy_mask, x, r);

  // TODO memcpy noise map to alpha channel.
  //plane_ptr->MemCpyOut(row.writable(z) + row_bnds.x1(), row_bnds.GetWidth() * sizeof(float), row_bnds);
  }
}

