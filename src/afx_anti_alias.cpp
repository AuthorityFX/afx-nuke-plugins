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
static const char* CLASS = "AFXAntiAlias";
static const char* HELP = "Anti Alias";

#define ThisClass AFXAntiAlias

using namespace DD::Image;

class ThisClass : public Iop {
private:

  // members to store knob values
  float k_threshold_;

  // members to store processed knob values

  boost::mutex mutex_;
  bool first_time_GPU_;
  bool first_time_CPU_;
  Lock lock_;

  afx::Bounds req_bnds_, format_bnds, format_f_bnds_;
  float proxy_scale_;

  afx::CudaProcess cuda_process_;
  afx::CudaImageArray row_c_imgs_;
  afx::CudaImageArray in_c_imgs_;
  afx::CudaStreamArray streams_;

  afx::ImageArray out_imgs_;

  afx::Threader threader_;

  void MetricsCPU(afx::Bounds region, const ImagePlane& source, const ImagePlane& matte, float* ref_hsv, double* sum_rgb, double* sum, double* sum_sqrs, unsigned int& num);
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
  inputs(1);

  first_time_GPU_ = true;
  first_time_CPU_ = true;

  try {
    cuda_process_.MakeReady();
  } catch (cudaError err) {}

  //initialize knobs
  k_threshold_ = 0.05f;
}
void ThisClass::knobs(Knob_Callback f) {

  Float_knob(f, &k_threshold_, "threshold", "Threshold");
  Tooltip(f, "Anti Alias Threshold");
  SetRange(f, 0.0, 1.0);

}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static Iop* build(Node* node) { return (new NukeWrapper(new ThisClass(node)))->channelsRGBoptionalAlpha(); }
const Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Anti Alias", build);

void ThisClass::_validate(bool) {
  copy_info(0);
  req_bnds_.SetBounds(info_.x(), info_.y(), info_.r() - 1, info_.t() - 1);
  format_bnds.SetBounds(input(0)->format().x(), input(0)->format().y(), input(0)->format().r() - 1, input(0)->format().t() - 1);
  format_f_bnds_.SetBounds(input(0)->full_size_format().x(), input(0)->full_size_format().y(), input(0)->full_size_format().r() - 1,
    input(0)->full_size_format().t() - 1);
  proxy_scale_ = (float)format_bnds.GetWidth() / (float)format_f_bnds_.GetWidth();
}
void ThisClass::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
  //Request source
  input0().request(x, y, r, t, channels, count);
}
void ThisClass::_open() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  out_imgs_.Clear();
}
void ThisClass::_close() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  out_imgs_.Clear();
}
void ThisClass::engine(int y, int x, int r, ChannelMask channels, Row& row) {
  callCloseAfter(0);
  ProcessCPU(y, x, r, channels, row); // TODO try catch, memset to prevent crash.
}
void ThisClass::ProcessCUDA(int y, int x, int r, ChannelMask channels, Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);
}
void ThisClass::ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);

  if (first_time_CPU_) {
    Guard guard(lock_);
    if (first_time_CPU_) {
      ImagePlane source_plane(Box(req_bnds_.x1(), req_bnds_.y1(), req_bnds_.x2() + 1, req_bnds_.y2() + 1), false, channels); // Create plane "false" = non-packed.
      if (aborted()) { return; } // return if aborted
      input0().fetchPlane(source_plane); // Fetch plane
      afx::Image temp(req_bnds_);
      foreach (z, source_plane.channels()) { // For each channel in plane
        temp.MemCpyIn(&source_plane.readable()[source_plane.chanNo(z) * source_plane.chanStride()], source_plane.rowStride() * sizeof(float), temp.GetBounds());
        out_imgs_.AddImage(req_bnds_);
        out_imgs_.GetBackPtr()->AddAttribute("channel", z);
        out_imgs_.GetBackPtr()->MemCpyIn(temp.GetPtr(), temp.GetPitch(), temp.GetBounds());
        afx::MorphAA aa;
        aa.Process(temp, *out_imgs_.GetBackPtr(), k_threshold_, threader_);
      }
      first_time_CPU_ = false;
    }
  } // End first time guard

  if (aborted()) { return; }

  foreach (z, channels) {
    afx::Image* plane_ptr = out_imgs_.GetPtrByAttribute("channel", z);
    plane_ptr->MemCpyOut(row.writable(z) + row_bnds.x1(), row_bnds.GetWidth() * sizeof(float), row_bnds);
  }
}
