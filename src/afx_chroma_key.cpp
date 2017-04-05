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

#include "include/threading.h"
#include "include/pixel.h"
#include "include/nuke_helper.h"
#include "include/median.h"
#include "include/color_op.h"

#define ThisClass AFXChromaKey

namespace nuke = DD::Image;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<float, 2, boost::geometry::cs::cartesian> Point;
typedef bg::model::multi_point<Point> PointCloud;
typedef bg::model::polygon<Point> Polygony;  // MSVC was complaining abour redefinition of "Polygon"
typedef bg::model::multi_polygon<Polygony> MultiGon;
typedef bg::model::box<Point> BoundingBox;

enum inputs {
  iSource = 0,
  iScreen = 1,
  iOutMatte = 2,
  iInMatte = 3,
};

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXChromaKey";
static const char* HELP = "Chroma Keyer";

class ThisClass : public nuke::Iop {
 private:
  // members to store knob values
  float k_out_dilate_;
  float k_in_dilate_;
  float k_out_falloff_;
  float k_in_falloff_;
  bool k_premultiply_;

  float out_falloff_;
  float in_falloff_;
  float out_dilate_;
  float in_dilate_;

  PointCloud out_pc_;
  PointCloud in_pc_;
  Polygony out_hull_;
  Polygony in_hull_;
  MultiGon out_hull_buffer_;
  MultiGon in_hull_buffer_;
  BoundingBox out_box_;
  BoundingBox in_box_;

  boost::mutex mutex_;
  bool first_time_engine_;
  nuke::Lock lock_;

  afx::Bounds req_bnds_, format_bnds_, format_f_bnds_;
  float proxy_scale_;

  afx::ImageThreader threader_;

  int maximum_inputs() const { return 4; }
  int minimum_inputs() const { return 4; }

  void Metrics(const afx::Bounds& region, const nuke::ImagePlane& source, const nuke::ImagePlane& out_matte, const nuke::ImagePlane& in_matte);

 public:
  explicit ThisClass(Node* node);
  void knobs(nuke::Knob_Callback);
  const char* Class() const;
  const char* node_help() const;
  static const Iop::Description d;
  Op* default_input(int input) const;
  const char* input_label(int input, char* buffer) const;

  void _validate(bool);
  void _request(int x, int y, int r, int t, nuke::ChannelMask channels, int count);
  void _open();
  void _close();
  void engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);
};
ThisClass::ThisClass(Node* node) : nuke::Iop(node) {
  first_time_engine_ = true;

  // initialize knobs
  k_out_dilate_ = 0.25f;
  k_out_falloff_ = 0.5f;
  k_in_dilate_ = 0.25f;
  k_in_falloff_ = 0.5f;
  k_premultiply_ = true;
}
void ThisClass::knobs(nuke::Knob_Callback f) {
  nuke::Float_knob(f, &k_out_dilate_, "out_dilate", "Out Dilate");
  nuke::Tooltip(f, "Dialte out region");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Float_knob(f, &k_out_falloff_, "out_falloff", "Out Falloff");
  nuke::Tooltip(f, "Out Matte Falloff");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Divider(f, "");

  nuke::Float_knob(f, &k_in_dilate_, "in_dilate", "In Dilate");
  nuke::Tooltip(f, "Dialte in region");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Float_knob(f, &k_in_falloff_, "in_falloff", "In Falloff");
  nuke::Tooltip(f, "In Matte Falloff");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Divider(f, "");

  nuke::Bool_knob(f, &k_premultiply_, "premultiply", "Premultiply");
  nuke::SetFlags(f, nuke::Knob::STARTLINE);
  nuke::Tooltip(f, "Premultiply by alpha");
}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static nuke::Iop* build(Node* node) { return new ThisClass(node); }
const nuke::Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Chroma Key", build);
nuke::Op* ThisClass::default_input(int input) const {
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
      input_longlabel(input) = "Screen Image";
      return "Screen";
    }
    case 2: {
      input_longlabel(input) = "Chroma values to key";
      return "Out";
    }
    case 3: {
      input_longlabel(input) = "Chroma values to keep";
      return "In";
    }
    default: {
      return 0;
    }
  }
}
void ThisClass::_validate(bool) {
  copy_info(0);
  if (input(iSource) != default_input(iSource)) {
    format_bnds_ = afx::BoxToBounds(input(0)->format());
    format_f_bnds_ = afx::BoxToBounds(input(0)->full_size_format());
    proxy_scale_ = static_cast<float>(format_bnds_.GetWidth()) / static_cast<float>(format_f_bnds_.GetWidth());

    out_falloff_ = fmaxf(1.0f + (1.0f - afx::math::Clamp(k_out_falloff_, 0.0f, 1.0f)), 0.008f);
    in_falloff_ = afx::math::Clamp(1.0f + k_in_falloff_, 0.008f, 125.0f);
    out_dilate_ = 50.0f * afx::math::Clamp(k_out_dilate_, 0.0f, 1.0f);
    in_dilate_ = 50.0f * afx::math::Clamp(k_in_dilate_, 0.0f, 1.0f);

    nuke::ChannelSet add_channels = channels();
    add_channels += nuke::Mask_RGBA;
    set_out_channels(add_channels);
    info_.turn_on(add_channels);

    info_.black_outside(true);
  } else {
    set_out_channels(nuke::Mask_None);
  }
}
void ThisClass::_request(int x, int y, int r, int t, nuke::ChannelMask channels, int count) {
  input(iSource)->request(input(iSource)->info().box(), channels, nuke::Mask_RGB);
  if (input(iScreen) != nullptr) { input(iScreen)->request(input(iScreen)->info().box(), nuke::Mask_RGB, count); }
  if (input(iOutMatte) != nullptr) { input(iOutMatte)->request(nuke::Mask_Alpha, count); }
  if (input(iInMatte) != nullptr) { input(iInMatte)->request(nuke::Mask_Alpha, count); }

  req_bnds_.SetBounds(x, y, r - 1, t - 1);
}
void ThisClass::_open() {
  first_time_engine_ = true;
}
void ThisClass::_close() {
  first_time_engine_ = true;
}
void ThisClass::engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  callCloseAfter(0);

  if (first_time_engine_) {
    nuke::Guard guard(lock_);
    if (first_time_engine_) {
      bool use_screen_input = false;
      afx::Bounds screen_bnds;
      if (input(iScreen)) {
        if (input(iSource)->info().box().intersects(input(iScreen)->info().box())) {
          screen_bnds = afx::InputBounds(input(iScreen));
          use_screen_input = true;
        }
      } else {
        screen_bnds = afx::InputBounds(input(iSource));
      }

      nuke::ImagePlane source_plane(afx::BoundsToBox(screen_bnds), false, nuke::Mask_RGB);  // Create plane "false" = non-packed.
      nuke::ImagePlane out_matte_plane(afx::BoundsToBox(screen_bnds), false, nuke::Mask_Alpha);  // Create plane "false" = non-packed.
      nuke::ImagePlane in_matte_plane(afx::BoundsToBox(screen_bnds), false, nuke::Mask_Alpha);  // Create plane "false" = non-packed.

      if (use_screen_input) {
        input(iScreen)->fetchPlane(source_plane);
      } else {
        input(iSource)->fetchPlane(source_plane);
      }
      if (input(iOutMatte) != nullptr) { input(iOutMatte)->fetchPlane(out_matte_plane); }
      if (input(iInMatte) != nullptr) { input(iInMatte)->fetchPlane(in_matte_plane); }

      out_pc_.clear();
      in_pc_.clear();
      threader_.ThreadRowChunks(screen_bnds, boost::bind(&ThisClass::Metrics, this, _1, boost::ref(source_plane), boost::ref(out_matte_plane), boost::ref(in_matte_plane)));
      threader_.Wait();

      out_hull_.clear();
      bg::convex_hull(out_pc_, out_hull_);
      in_hull_.clear();
      bg::convex_hull(in_pc_, in_hull_);

      // Declare strategies
      bg::strategy::buffer::distance_symmetric<double> out_distance_strategy(out_dilate_);
      bg::strategy::buffer::distance_symmetric<double> in_distance_strategy(in_dilate_);
      bg::strategy::buffer::join_miter join_strategy;
      bg::strategy::buffer::end_flat end_strategy;
      bg::strategy::buffer::point_square point_strategy;
      bg::strategy::buffer::side_straight side_strategy;

      out_hull_buffer_.clear();
      bg::buffer(out_hull_, out_hull_buffer_, out_distance_strategy, side_strategy, join_strategy, end_strategy, point_strategy);
      in_hull_buffer_.clear();
      bg::buffer(in_hull_, in_hull_buffer_, in_distance_strategy, side_strategy, join_strategy, end_strategy, point_strategy);

      bg::envelope(out_hull_buffer_, out_box_);
      bg::envelope(in_hull_buffer_, in_box_);

      first_time_engine_ = false;
    }
  }  // End first time guard


  nuke::Channel rgba_chan[4] = {nuke::Chan_Red, nuke::Chan_Green, nuke::Chan_Blue, nuke::Chan_Alpha};

  // Remove RGBA from channels and copy all other channels straight through to output row
  nuke::ChannelSet copy_mask = channels - nuke::Mask_RGBA;
  row.get(input0(), y, x, r, copy_mask);
  row.pre_copy(row, copy_mask);
  row.copy(row, copy_mask, x, r);

  // Must call get before initializing any pointers
  row.get(input0(), y, x, r, nuke::Mask_RGBA);

  afx::Pixel<const float> in(3);
  afx::Pixel<float> out(4);
  for (int i = 0; i < 3; ++i) { in.SetPtr(row[rgba_chan[i]] + x, i); }  // Set const in pointers RGB
  for (int i = 0; i < 4; ++i) { out.SetPtr(row.writable(rgba_chan[i]) + x, i); }  // Set out pointers RGBA

  for (int x0 = x; x0 < r; ++x0) {  // Loop through pixels in row
    float rgb[3];
    float lab[3];
    for (int i = 0; i < 3; ++i) { rgb[i] = in.GetVal(i); }
    afx::RGBtoLab(rgb, lab);
    Point p = Point(lab[1], lab[2]);
    float matte = 1.0f;
    if (bg::within(p, out_box_)) {
      if (bg::within(p, out_hull_buffer_)) {
        if (bg::within(p, out_hull_)) {
          matte = 0;
        } else {
          matte = 1.0f - powf(out_falloff_, -bg::distance(p, out_hull_));
          if (input(iInMatte) != nullptr) {
            if (bg::within(p, in_box_)) {
              if (bg::within(p, in_hull_buffer_)) {
                if (bg::within(p, in_hull_)) {
                  matte = 1.0f;
                } else {
                  matte = fmaxf(matte, powf(in_falloff_, -bg::distance(p, in_hull_)));
                }
              }
            }
          }
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

void ThisClass::Metrics(const afx::Bounds& region, const nuke::ImagePlane& source, const nuke::ImagePlane& out_matte, const nuke::ImagePlane& in_matte) {
  afx::Bounds plane_bnds(afx::BoxToBounds(source.bounds()));

  float rgb[3];
  PointCloud out_pc;
  PointCloud in_pc;

  afx::Pixel<const float> in(3);
  const float zero = 0.0f;
  const float* out_m_ptr = &zero;
  const float* in_m_ptr = &zero;

  nuke::Channel chan_rgb[3] = {nuke::Chan_Red, nuke::Chan_Green, nuke::Chan_Blue};

  for (int y = region.y1(); y <= region.y2(); ++y) {
    for  (int i = 0; i < 3; ++i) {
      in.SetPtr(&source.readable()[source.chanNo(chan_rgb[i]) * source.chanStride() + source.rowStride() * (plane_bnds.ClampY(y) - plane_bnds.y1()) +
                                   plane_bnds.ClampX(region.x1()) - plane_bnds.x1()], i);
    }
    if (input(iOutMatte) != nullptr) {
      out_m_ptr = &out_matte.readable()[out_matte.chanNo(nuke::Chan_Alpha) * out_matte.chanStride() + out_matte.rowStride() *
                                        (plane_bnds.ClampY(y) - plane_bnds.y1()) + plane_bnds.ClampX(region.x1()) - plane_bnds.x1()];
    }
    if (input(iInMatte) != nullptr) {
      in_m_ptr = &in_matte.readable()[in_matte.chanNo(nuke::Chan_Alpha) * in_matte.chanStride() + in_matte.rowStride() *
                                      (plane_bnds.ClampY(y) - plane_bnds.y1()) + plane_bnds.ClampX(region.x1()) - plane_bnds.x1()];
    }

    for (int x = region.x1(); x <= region.x2(); ++x) {
      for (int i = 0; i < 3; i++) { rgb[i] = in.GetVal(i); }
      Point p;
      if (*out_m_ptr > 0.5 || *in_m_ptr > 0.5) {
        float lab[3];
        afx::RGBtoLab(rgb, lab);
        p = Point(lab[1], lab[2]);
      }
      if (*out_m_ptr > 0.5) {
        out_pc.push_back(p);
      }
      if (*in_m_ptr > 0.5) {
        in_pc.push_back(p);
      }
      in.NextPixel();
      if (input(iOutMatte) != nullptr) { out_m_ptr++; }
      if (input(iInMatte) != nullptr) { in_m_ptr++; }
    }
  }

  boost::mutex::scoped_lock lock(mutex_);
  bg::append(out_pc_, out_pc);
  bg::append(in_pc_, in_pc);
}

