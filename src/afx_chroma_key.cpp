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

#include <math.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>

#include "threading.h"
#include "median.h"
#include "color_op.h"

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXChromaKey";
static const char* HELP = "Chroma Keyer";

#define ThisClass AFXChromaKey

enum inputs
{
  iSource = 0,
  iSourceAtTime = 1,
  iScreenMatte = 2,
  iScreenMatteAtTime = 3,
};

using namespace DD::Image;

class ThisClass : public Iop {
private:

  // members to store knob values
  float k_screen_color_[3];
  bool k_specify_source_frame_;
  float k_source_frame_;

  float k_chroma_clean_;
  float k_luma_clean_;
  float k_falloff_;
  bool k_premultiply_;

  float k_screen_mean_[3];

  float mean_rgb_[3];
  float mean_lab_[3];
  float std_dev_lab_[3];

  // members to store processed knob values
  float smoothness_;

  boost::mutex mutex_;
  bool first_time_engine_;
  Lock lock_;

  afx::Bounds req_bnds_, format_bnds, format_f_bnds_;
  float proxy_scale_;

  afx::Threader threader_;

  int maximum_inputs() const { return 2; }
  int minimum_inputs() const { return 2; }

  void MetricsCPU(const afx::Bounds& region, const ImagePlane& source, const ImagePlane& matte, float hue);

public:
  ThisClass(Node* node);
  void knobs(Knob_Callback);
  int knob_changed(Knob* k);
  const char* Class() const;
  const char* node_help() const;
  static const Iop::Description d;
  Op* default_input(int input) const;
  const char* input_label(int input, char* buffer) const;
  int split_input(int n) const;
  const OutputContext& inputContext(int n, int split, OutputContext& context) const;

  void _validate(bool);
  void _request(int x, int y, int r, int t, ChannelMask channels, int count);
  void _open();
  void _close();
  void engine(int y, int x, int r, ChannelMask channels, Row& row);
};
ThisClass::ThisClass(Node* node) : Iop(node) {

  first_time_engine_ = true;

  //initialize knobs
  k_screen_color_[0] = k_screen_color_[2] = 0;
  k_screen_color_[1] = 1;
  k_specify_source_frame_ = false;
  k_source_frame_ = 1.0f;

  k_chroma_clean_ = 0.5f;
  k_luma_clean_ = 0.5f;
  k_falloff_ = 0.5f;
  k_premultiply_ = true;
}
void ThisClass::knobs(Knob_Callback f) {
  Color_knob(f, k_screen_color_, "screen_color", "Screen Color");
  Tooltip(f, "Approximate color of chroma screen");
  ClearFlags(f, Knob::MAGNITUDE | Knob::SLIDER);

  Bool_knob(f, &k_specify_source_frame_, "specify_source_frame", "Specify Source Frame");
  Tooltip(f, "Specify source frame for screen metrics");
  SetFlags(f, Knob::EARLY_STORE | Knob::KNOB_CHANGED_ALWAYS);

  Float_knob(f, &k_source_frame_, "source_frame", "Source Frame");
  Tooltip(f, "Source frame for screen metrics");
  SetRange(f, 0.0, 150.0);
  SetFlags(f, Knob::STARTLINE | Knob::HIDDEN | Knob::EARLY_STORE);

  Float_knob(f, &k_chroma_clean_, "chrome_clean", "Chroma Clean");
  Tooltip(f, "Cleanup screen");
  SetRange(f, 0.0, 1.0);

  Float_knob(f, &k_luma_clean_, "luma_clean", "Luma Clean");
  Tooltip(f, "Cleanup screen");
  SetRange(f, 0.0, 1.0);

  Float_knob(f, &k_falloff_, "falloff", "Falloff");
  Tooltip(f, "Matte Falloff");
  SetRange(f, 0.0, 1.0);

  Bool_knob(f, &k_premultiply_, "premultiply", "Premultiply");
  SetFlags(f, Knob::STARTLINE);
  Tooltip(f, "Premultiply by alpha");

}
int ThisClass::knob_changed(Knob* k) {
  if (k->is("specify_source_frame")) {
    knob("source_frame")->visible(knob("specify_source_frame")->get_value() >= 0.5);
    return 1;
  }
  return Iop::knob_changed(k);
}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static Iop* build(Node* node) { return new ThisClass(node); }
const Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Chroma Key", build);
Op* ThisClass::default_input(int input) const {
  if (input == 0) { return Iop::default_input(0); }
  return 0;
}
const char* ThisClass::input_label(int input, char* buffer) const {
  switch (input) {
    case 0: {
      input_longlabel(input) = "Source Image";
      return "Source";
    }
    case 1: {
      input_longlabel(input) = "Screen Matte";
      return "Screen Matte";
    }
    default: {
      return 0;
    }
  }
}
int ThisClass::split_input(int input_num) const {
  if (input_num == 0 || input_num == 1) {
    return 2;
  } else {
    return 1;
  }
}
const OutputContext& ThisClass::inputContext(int n, int split, OutputContext& oc) const {
  oc = outputContext();
  if (split == 1) { oc.setFrame(k_source_frame_); }
  return oc;
}
void ThisClass::_validate(bool) {
  if (input(iSource) != default_input(iSource)) {
    //Copy source info
    copy_info(0);
    req_bnds_.SetBounds(info_.x(), info_.y(), info_.r() - 1, info_.t() - 1);
    format_bnds.SetBounds(input(0)->format().x(), input(0)->format().y(), input(0)->format().r() - 1, input(0)->format().t() - 1);
    format_f_bnds_.SetBounds(input(0)->full_size_format().x(), input(0)->full_size_format().y(), input(0)->full_size_format().r() - 1,
      input(0)->full_size_format().t() - 1);
    proxy_scale_ = (float)format_bnds.GetWidth() / (float)format_f_bnds_.GetWidth();

    ChannelSet add_channels = channels();
    add_channels += Chan_Red;
    add_channels += Chan_Green;
    add_channels += Chan_Blue;
    add_channels += Chan_Alpha;
    set_out_channels(add_channels);
    info_.turn_on(add_channels);

    info_.black_outside(true);
    // Process knob values
    smoothness_ = k_falloff_ * 10.f;

  } else {
    set_out_channels(Mask_None);
    copy_info();
  }
}
void ThisClass::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
  //Request source
  if (input(iSource) != default_input(iSource)) {
    input(iSource)->request(channels, count);
    input(iSourceAtTime)->request(channels, count);
  }
  //Request screen matte
  if (input(iScreenMatte) != nullptr) {
    input(iScreenMatte)->request(channels, count);
    input(iScreenMatteAtTime)->request(channels, count);
  }
}
void ThisClass::_open() {
  first_time_engine_ = true;
}
void ThisClass::_close() {
  first_time_engine_ = true;
}
void ThisClass::engine(int y, int x, int r, ChannelMask channels, Row& row) {
  callCloseAfter(0);

  if (first_time_engine_) {
    Guard guard(lock_);
    if (first_time_engine_) {
      Box plane_req_box(info_.box());
      ImagePlane source_plane(plane_req_box, false, Mask_RGB); // Create plane "false" = non-packed.
      ImagePlane matte_plane(plane_req_box, false, Mask_Alpha); // Create plane "false" = non-packed.
      if (k_specify_source_frame_) {
        input(iSourceAtTime)->fetchPlane(source_plane);
        if (input(iScreenMatte) != nullptr) { input(iScreenMatteAtTime)->fetchPlane(matte_plane); }
      } else {
        input(iSource)->fetchPlane(source_plane);
        if (input(iScreenMatte) != nullptr) { input(iScreenMatte)->fetchPlane(matte_plane); }
      }

      float ref_hue = afx::Hue(k_screen_color_);
      threader_.ThreadImageChunks(format_bnds, boost::bind(&ThisClass::MetricsCPU, this, _1, boost::ref(source_plane), boost::ref(matte_plane), ref_hue));
      threader_.Synchonize();


      first_time_engine_ = false;
    }
  } // End first time guard


  Channel rgba_chan[4];
  rgba_chan[0] = Chan_Red;
  rgba_chan[1] = Chan_Green;
  rgba_chan[2] = Chan_Blue;
  rgba_chan[3] = Chan_Alpha;

  // Remove RGBA from channels and copy all other channels straight through to output row
  ChannelSet copy_mask = channels - Mask_RGBA;
  row.get(input0(), y, x, r, channels);
  row.pre_copy(row, copy_mask);
  row.copy(row, copy_mask, x, r);

  //Must call get before initializing any pointers
  row.get(input0(), y, x, r, Mask_RGBA);

  float lab[3];
  float hsv[3];
  float std_lab[3];
  float s_res, v_res;
  float rad, matte;

  afx::ReadOnlyPixel in(3);
  afx::Pixel out(4);
  for (int i = 0; i < 3; ++i) { in.SetPtr(row[rgba_chan[i]] + x, i); } // Set const in pointers RGB
  for (int i = 0; i < 4; ++i) { out.SetPtr(row.writable(rgba_chan[i]) + x, i); } // Set out pointers RGBA

  for (int x0 = x; x0 < r; ++x0) { // Loop through pixels in row
    for (int i = 0; i < 3; ++i) { lab[i] = hsv[i] = in.GetVal(i); }


    for (int i = 0; i < 3; ++i) {
      if (k_premultiply_) {
        out.SetVal(in.GetVal(i) * matte, i);
      } else {
        out.SetVal(in.GetVal(i), i);
      }
    }
    out.SetVal(matte, 3);
    in++;
    out++;
  }
}

void ThisClass::MetricsCPU(const afx::Bounds& region, const ImagePlane& source, const ImagePlane& matte, float ref_hue) {

  afx::Bounds plane_bnds(source.bounds().x(), source.bounds().y(), source.bounds().r() - 1, source.bounds().t() - 1);

  float rgb[3];

  typedef boost::geometry::model::point<float, 2, boost::geometry::cs::cartesian> point;

  afx::ReadOnlyPixel in(3);
  const float one = 1.0f;
  const float* m_ptr = &one;

  Channel chan_rgb[3] = {Chan_Red, Chan_Green, Chan_Blue};

  for (int y = region.y1(); y <= region.y2(); y+=4) {
    for  (int i = 0; i < 3; ++i) {
      in.SetPtr(&source.readable()[source.chanNo(chan_rgb[i]) * source.chanStride() + source.rowStride() * (plane_bnds.ClampY(y) - plane_bnds.y1()) +
      plane_bnds.ClampX(region.x1()) - plane_bnds.x1()], i);
    }
    if (input(iScreenMatte) != nullptr) { m_ptr =
                                                  &matte.readable()[matte.chanNo(Chan_Alpha) * matte.chanStride() + matte.rowStride() *
                                                  (plane_bnds.ClampY(y) - plane_bnds.y1()) + plane_bnds.ClampX(region.x1()) - plane_bnds.x1()];
                                        }

    for (int x = region.x1(); x <= region.x2(); x+=4) {
      //Convert HSV and convert pixel to HSV
      for (int i = 0; i < 3; i++) { rgb[i] = in.GetVal(i); }

      if (*m_ptr > 0.5) { //Compute stats from screen
        float hue = afx::Hue(rgb);
        if (fabsf(hue - ref_hue) < 0.1f ) {

          float lab[3];
          afx::RGBtoLab(rgb, lab);

          //Add to point cloud

        }
      }
      for (int i = 0; i < 4; i++) { in.NextPixel(); }
      if (input(iScreenMatte) != nullptr) {
        for (int i = 0; i < 4; i++) { m_ptr++; }
      }
    }
  }


  boost::mutex::scoped_lock lock(mutex_);
  // Add to main geo


}
