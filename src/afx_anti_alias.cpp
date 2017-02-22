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

#include "threading.h"
#include "image.h"
#include "nuke_helper.h"
#include "mlaa.h"

#define ThisClass AFXAntiAlias

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXAntiAlias";
static const char* HELP = "Anti Alias";

using namespace DD::Image;

class ThisClass : public Iop {
 private:
  // members to store knob values
  float k_threshold_;

  // members to store processed knob values
  boost::mutex mutex_;
  bool first_time_CPU_;
  Lock lock_;

  afx::Bounds req_bnds_, format_bnds_, format_f_bnds_;
  float proxy_scale_;

  afx::ImageArray out_imgs_;

  afx::Threader threader_;

  void ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row);

 public:
  explicit ThisClass(Node* node);
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

  first_time_CPU_ = true;

  // initialize knobs
  k_threshold_ = 0.25f;
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

  format_bnds_ = afx::BoxToBounds(input(0)->format());
  format_f_bnds_ = afx::BoxToBounds(input(0)->full_size_format());
  proxy_scale_ = static_cast<float>(format_bnds_.GetWidth()) / static_cast<float>(format_f_bnds_.GetWidth());
}
void ThisClass::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
  // Request source
  Box req_box(x + 50, y + 50, r + 50, t + 50);  // TODO(rpw): add knob to control this
  input0().request(req_box, channels, count);
  req_bnds_.SetBounds(x, y, r - 1, t - 1);
}
void ThisClass::_open() {
  first_time_CPU_ = true;
}
void ThisClass::_close() {
  first_time_CPU_ = true;
  out_imgs_.Clear();
}
void ThisClass::engine(int y, int x, int r, ChannelMask channels, Row& row) {
  callCloseAfter(0);
  ProcessCPU(y, x, r, channels, row);
}
void ThisClass::ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);

  {
    Guard guard(lock_);
    if (first_time_CPU_) {
      afx::Bounds req_pad_bnds = req_bnds_.GetPadBounds(50);
      req_pad_bnds.Intersect(afx::InputBounds(input(0)));

      ImagePlane source_plane(afx::BoundsToBox(req_pad_bnds), false, channels);  // Create plane "false" = non-packed.
      input0().fetchPlane(source_plane);  // Fetch plane
      out_imgs_.Clear();
      afx::Image in_img(req_pad_bnds);
      foreach(z, source_plane.channels()) {  // For each channel in plane
        // TODO(rpw): don't need to copy this image, just need to setup pointer, pitch, bounds
        in_img.MemCpyIn(&source_plane.readable()[source_plane.chanNo(z) * source_plane.chanStride()], source_plane.rowStride() * sizeof(float), in_img.GetBounds());
        out_imgs_.AddImage(req_pad_bnds);
        out_imgs_.GetBackPtr()->AddAttribute("channel", z);
        afx::MorphAA aa;
        aa.Process(in_img, out_imgs_.GetBackPtr(), k_threshold_, &threader_);
      }
      first_time_CPU_ = false;
    }
  }  // End first time guard

  if (aborted()) { return; }

  foreach(z, channels) {
    afx::Image* chan_ptr = out_imgs_.GetPtrByAttribute("channel", z);
    chan_ptr->MemCpyOut(row.writable(z) + row_bnds.x1(), row_bnds.GetWidth() * sizeof(float), row_bnds);
  }
}
