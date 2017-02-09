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

#include "threading.h"
#include "median.h"
#include "color_op.h"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXChromaKey";
static const char* HELP = "Chroma Keyer";

#define ThisClass AFXChromaKey

typedef bg::model::point<float, 2, boost::geometry::cs::cartesian> Point;
typedef bg::model::multi_point<Point> PointCloud;
typedef bg::model::polygon<Point> Polygon;
typedef bg::model::multi_polygon<Polygon> MultiGon;
typedef bg::model::box<Point> BoundingBox;

enum inputs
{
  iSource = 0,
  iScreenMatte = 1,
};

using namespace DD::Image;

class ThisClass : public Iop {
private:

  // members to store knob values
  float k_smoothness_;
  float k_falloff_;
  bool k_premultiply_;

  PointCloud screen_pc_;
  Polygon screen_hull_;
  MultiGon screen_hull_offset_;
  BoundingBox screen_box_;

  boost::mutex mutex_;
  bool first_time_engine_;
  Lock lock_;

  afx::Bounds req_bnds_, format_bnds, format_f_bnds_;
  float proxy_scale_;

  afx::Threader threader_;

  int maximum_inputs() const { return 2; }
  int minimum_inputs() const { return 2; }

  void MetricsCPU(const afx::Bounds& region, const ImagePlane& source, const ImagePlane& matte, int step);

public:
  ThisClass(Node* node);
  void knobs(Knob_Callback);
  const char* Class() const;
  const char* node_help() const;
  static const Iop::Description d;
  Op* default_input(int input) const;
  const char* input_label(int input, char* buffer) const;

  void _validate(bool);
  void _request(int x, int y, int r, int t, ChannelMask channels, int count);
  void _open();
  void _close();
  void engine(int y, int x, int r, ChannelMask channels, Row& row);
};
ThisClass::ThisClass(Node* node) : Iop(node) {

  first_time_engine_ = true;

  //initialize knobs
  k_smoothness_ = 0.5f;
  k_falloff_ = 0.5f;
  k_premultiply_ = true;
}
void ThisClass::knobs(Knob_Callback f) {
  Float_knob(f, &k_smoothness_, "smoothness", "Smoothness");
  Tooltip(f, "Smoothness of matte");
  SetRange(f, 0.0, 1.0);

  Float_knob(f, &k_falloff_, "falloff", "Falloff");
  Tooltip(f, "Matte Falloff");
  SetRange(f, 0.0, 1.0);

  Bool_knob(f, &k_premultiply_, "premultiply", "Premultiply");
  SetFlags(f, Knob::STARTLINE);
  Tooltip(f, "Premultiply by alpha");

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

  } else {
    set_out_channels(Mask_None);
    copy_info();
  }
}
void ThisClass::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
  //Request source
  input(iSource)->request(channels, count);
  //Request screen matte
  if (input(iScreenMatte) != nullptr) { input(iScreenMatte)->request(channels, count); }

  req_bnds_.SetBounds(x, y, r - 1, t - 1);
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

      input(iSource)->fetchPlane(source_plane);
      if (input(iScreenMatte) != nullptr) { input(iScreenMatte)->fetchPlane(matte_plane); }

      screen_pc_.clear();
      screen_hull_.clear();

      threader_.ThreadImageChunks(format_bnds, boost::bind(&ThisClass::MetricsCPU, this, _1, boost::ref(source_plane), boost::ref(matte_plane), 4));
      threader_.Synchonize();

      bg::convex_hull(screen_pc_, screen_hull_);

      // Declare strategies
      bg::strategy::buffer::distance_symmetric<double> distance_strategy(100.0 * k_smoothness_);
      bg::strategy::buffer::join_miter join_strategy;
      bg::strategy::buffer::end_flat end_strategy;
      bg::strategy::buffer::point_square point_strategy;
      bg::strategy::buffer::side_straight side_strategy;

      bg::buffer(screen_hull_, screen_hull_offset_, distance_strategy, side_strategy, join_strategy, end_strategy, point_strategy);
      bg::envelope(screen_hull_offset_, screen_box_);

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

  afx::ReadOnlyPixel in(3);
  afx::Pixel out(4);
  for (int i = 0; i < 3; ++i) { in.SetPtr(row[rgba_chan[i]] + x, i); } // Set const in pointers RGB
  for (int i = 0; i < 4; ++i) { out.SetPtr(row.writable(rgba_chan[i]) + x, i); } // Set out pointers RGBA

  for (int x0 = x; x0 < r; ++x0) { // Loop through pixels in row
    float rgb[3];
    float lab[3];
    for (int i = 0; i < 3; ++i) { rgb[i] = in.GetVal(i); }
    afx::RGBtoLab(rgb, lab);
    Point p = Point(lab[1], lab[2]);
    float matte = 1.0f;
    if (bg::within(p, screen_box_)) {
      if (bg::within(p, screen_hull_offset_)) {
        if (bg::within(p, screen_hull_)) {
          matte = 0;
        } else {
          matte = 1.0f - pow((1.0f + k_falloff_), -bg::distance(p, screen_hull_));
        }
      }
    }

    if (k_premultiply_) {
      for (int i = 0; i < 3; ++i) { out.SetVal(in.GetVal(i) * matte, i); }
    } else {
      for (int i = 0; i < 3; ++i) { out.SetVal(in.GetVal(i), i); }
    }
    out.SetVal(matte, 3);
    in++;
    out++;
  }
}

void ThisClass::MetricsCPU(const afx::Bounds& region, const ImagePlane& source, const ImagePlane& matte, int step) {

  afx::Bounds plane_bnds(source.bounds().x(), source.bounds().y(), source.bounds().r() - 1, source.bounds().t() - 1);

  float rgb[3];
  PointCloud pc;

  afx::ReadOnlyPixel in(3);
  const float zero = 0.0f;
  const float* m_ptr = &zero;

  Channel chan_rgb[3] = {Chan_Red, Chan_Green, Chan_Blue};

  for (int y = region.y1(); y <= region.y2(); y += step) {
    for  (int i = 0; i < 3; ++i) {
      in.SetPtr(&source.readable()[source.chanNo(chan_rgb[i]) * source.chanStride() + source.rowStride() * (plane_bnds.ClampY(y) - plane_bnds.y1()) +
                plane_bnds.ClampX(region.x1()) - plane_bnds.x1()], i);
    }
    if (input(iScreenMatte) != nullptr) { m_ptr = &matte.readable()[matte.chanNo(Chan_Alpha) * matte.chanStride() + matte.rowStride() *
                                                  (plane_bnds.ClampY(y) - plane_bnds.y1()) + plane_bnds.ClampX(region.x1()) - plane_bnds.x1()];
                                        }
    for (int x = region.x1(); x <= region.x2(); x += step) {
      for (int i = 0; i < 3; i++) { rgb[i] = in.GetVal(i); }
      if (*m_ptr > 0.5) {
        float lab[3];
        afx::RGBtoLab(rgb, lab);
        Point p = Point(lab[1], lab[2]);
        Polygon hull;
        boost::geometry::convex_hull(pc, hull);
        if (!boost::geometry::within(p, hull)) {
          pc.push_back(p);
        }
      }
      for (int i = 0; i < step; i++) { in.NextPixel(); }
      if (input(iScreenMatte) != nullptr) {
        for (int i = 0; i < step; i++) { m_ptr++; }
      }
    }
  }

  boost::mutex::scoped_lock lock(mutex_);
  bg::append(screen_pc_, pc);

}

