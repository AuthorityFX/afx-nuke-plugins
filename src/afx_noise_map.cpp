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
#include <math.h>

#include "include/image_tools.h"

#include "include/threading.h"
#include "include/image.h"
#include "include/nuke_helper.h"
#include "include/noise_map.h"

#define ThisClass AFXNoiseMap

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXNoiseMap";
static const char* HELP = "Noise Map";

namespace nuke = DD::Image;

class ThisClass : public nuke::Iop {
private:
  // members to store knob values
  int k_size_;
  int k_level_;
  // members to store processed knob values
  boost::mutex mutex_;
  bool first_time_CPU_;
  nuke::Lock lock_;

  afx::Bounds req_bnds_, format_bnds_, format_f_bnds_;
  float proxy_scale_;

  afx::ImageArray out_imgs_;

  afx::ImageThreader threader_;

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
  inputs(1);

  first_time_CPU_ = true;

  // initialize knobs
  k_size_ = 3;
  k_level_ = 0;
}
void ThisClass::knobs(nuke::Knob_Callback f) {
  nuke::Int_knob(f, &k_size_, "size", "Size");
  nuke::Tooltip(f, "Window Size");
  nuke::SetFlags(f, nuke::Knob::SLIDER);
  nuke::SetRange(f, 0, 10);

  nuke::Int_knob(f, &k_level_, "level", "Level");
  nuke::Tooltip(f, "Wavelet Level");
  nuke::SetFlags(f, nuke::Knob::SLIDER);
  nuke::SetRange(f, 0, 10);
}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static nuke::Iop* build(Node* node) { return (new nuke::NukeWrapper(new ThisClass(node)))->channelsRGBoptionalAlpha(); }
const nuke::Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Noise Map", build);

void ThisClass::_validate(bool) {
  copy_info(0);

  format_bnds_ = afx::BoxToBounds(input(0)->format());
  format_f_bnds_ = afx::BoxToBounds(input(0)->full_size_format());
  proxy_scale_ = static_cast<float>(format_bnds_.GetWidth()) / static_cast<float>(format_f_bnds_.GetWidth());
}
void ThisClass::_request(int x, int y, int r, int t, nuke::ChannelMask channels, int count) {
  // Request source
  nuke::Box req_box(x, y, r, t);
  req_bnds_ = afx::BoxToBounds(req_box);
  req_box.pad(18);
  input0().request(req_box, channels, count);
}
void ThisClass::_open() {
  first_time_CPU_ = true;
}
void ThisClass::_close() {
  first_time_CPU_ = true;
  out_imgs_.Clear();
}
void ThisClass::engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  callCloseAfter(0);
  ProcessCPU(y, x, r, channels, row);
}
void ThisClass::ProcessCPU(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);
  {
    nuke::Guard guard(lock_);
    if (first_time_CPU_) {
      first_time_CPU_ = false;

      afx::Bounds req_pad_bnds = req_bnds_.GetPadBounds(50);
      afx::Bounds req_pad_bnds_clamped = req_pad_bnds;
      req_pad_bnds_clamped.Intersect(afx::InputBounds(input(0)));

      out_imgs_.Clear();
      afx::Image in_img(req_pad_bnds);
      foreach(z, channels) {
        if (aborted()) { return; }

        afx::FetchImage(&in_img, input(0), z, req_pad_bnds_clamped);
        out_imgs_.Add(req_bnds_);
        afx::Image* out_img = out_imgs_.GetBackPtr();
        out_img->AddAttribute("channel", z);

        afx::BorderExtender be;
        be.Mirror(&in_img, req_pad_bnds_clamped);

        afx::NoiseMap nm;
        nm.Calculate(in_img, out_img, k_size_);

      }
    }
  }  // End first time guard

  if (aborted()) { return; }
  foreach(z, channels) {
    afx::Image* chan_ptr = out_imgs_.GetPtrByAttribute("channel", z);
    chan_ptr->MemCpyOut(row.writable(z) + row_bnds.x1(), row_bnds.GetWidth() * sizeof(float), row_bnds);
  }
}
