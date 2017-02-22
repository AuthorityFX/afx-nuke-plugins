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

#include "include/nuke_helper.h"
#include "include/median.h"
#include "include/color_op.h"

#define ThisClass AFXDeSpill

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXDeSpill";
static const char* HELP = "Remove Spill";

enum inputs {
  iSource = 0,
  iMatte = 1
};

const char* algorithm_list[8] = {
  "Light",
  "Medium",
  "Hard",
  "Harder",
  "Even Harder",
  "Extreme",
  "Uber",
  0
};

using namespace DD::Image;

class ThisClass : public Iop {
 private:
  // members to store knob values
  float k_screen_color_[3];
  float k_amount_;
  float k_lightness_adj;
  int k_algorithm_;
  ChannelSet k_spill_matte_channel_;

  afx::RotateColor hue_shifter_;
  afx::RotateColor hue_shifter_inv_;
  float ref_suppression_;
  afx::ScreenColor color_;

  void ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row);

 public:
  explicit ThisClass(Node* node);
  void knobs(Knob_Callback);
  int knob_changed(Knob* k);
  const char* Class() const;
  const char* node_help() const;
  static const Iop::Description d;
  Op* default_input(int input) const;
  const char* input_label(int input, char* buffer) const;

  void _validate(bool);
  void _request(int x, int y, int r, int t, ChannelMask channels, int count);
  void engine(int y, int x, int r, ChannelMask channels, Row& row);
};
ThisClass::ThisClass(Node* node) : Iop(node) {
  // Set inputs
  inputs(2);

  // initialize knobs
  k_algorithm_ = afx::kHarder;
  k_amount_ = 1.0f;
  k_lightness_adj = 1.0f;
  k_screen_color_[0] = k_screen_color_[2] = 0.0f;
  k_screen_color_[1] = 1.0f;
}
void ThisClass::knobs(Knob_Callback f) {
  Color_knob(f, k_screen_color_, "screen_color", "Screen Color");
  Tooltip(f, "Color of Chroma Screen");
  ClearFlags(f, Knob::MAGNITUDE | Knob::SLIDER);

  Divider(f, "Settings");

  Float_knob(f, &k_amount_, "amount", "Spill Amount");
  Tooltip(f, "Amount of spill to remove");
  SetRange(f, 0.0, 1.0);
  SetFlags(f, Knob::FORCE_RANGE);

  Float_knob(f, &k_lightness_adj, "lightness_adj", "Lightness Adjustment");
  Tooltip(f, "Adjust perceived lightness");
  SetRange(f, 0.0, 1.0);
  SetFlags(f, Knob::FORCE_RANGE);

  Enumeration_knob(f, &k_algorithm_, algorithm_list, "algorithm", "Algorithm");
  Tooltip(f, "Spill Algorithm");

  Divider(f, "Output");

  ChannelSet_knob(f, &k_spill_matte_channel_, "spill_matte", "Spill Matte");
}
int ThisClass::knob_changed(Knob* k) {
  return Iop::knob_changed(k);
}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static Iop* build(Node* node) { return new ThisClass(node); }
const Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Smart Median", build);
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
      input_longlabel(input) = "Spill Holdout";
      return "Spill Holdout";
    }
    default: {
      return 0;
    }
  }
}
void ThisClass::_validate(bool) {
  copy_info(0);

  ChannelSet out_channels = channels();
  out_channels += k_spill_matte_channel_;
  set_out_channels(out_channels);
  info_.turn_on(out_channels);

  // 1/3 hue is green, 2/3 hue is blue. center_ chosen based on max(g, b)
  // (center_ - mean) * 360 is the angular distance in degrees from ideal screen hue
  float hsv[3];
  float ref_rgb[3];
  for (int i = 0; i < 3; i++) { ref_rgb[i] = k_screen_color_[i]; }
  afx::RGBtoHSV(ref_rgb, hsv);
  float center;
  if (ref_rgb[1] > ref_rgb[2]) {
    center = 1.0f/3.0f;
    color_ = afx::kGreen;
  } else if (ref_rgb[2] > ref_rgb[1]) {
    center = 2.0f/3.0f;
    color_ = afx::kBlue;
  }
  float hue_rotation = 360.0f * (center - hsv[0]);
  hue_shifter_.BuildMatrix(hue_rotation);  // Initialize hue shifter object
  hue_shifter_.Rotate(ref_rgb);  // Rotate hue of ref RGB so that the mean hue is pure green
  ref_suppression_ = afx::SpillSuppression(ref_rgb, k_algorithm_, color_);  // Spill suppressoin of ref_rgb
  hue_shifter_inv_ = hue_shifter_;
  hue_shifter_inv_.Invert();
}
void ThisClass::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
  ChannelSet req_channels = channels;
  req_channels += Mask_RGB;
  input(iSource)->request(x, y, r, t, req_channels, count);  // Only request RGB
  if (input(iMatte) != nullptr) { input(iMatte)->request(x, y, r, t, Mask_Alpha, count); }
}
void ThisClass::engine(int y, int x, int r, ChannelMask channels, Row& row) {
  callCloseAfter(0);
  ProcessCPU(y, x, r, channels, row);
}
void ThisClass::ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row) {
  ChannelSet req_channels = channels;
  req_channels += Mask_RGB;  // Add RGB to channels
  row.get(input0(), y, x, r, req_channels);  // Request all channels
  // Copy channels that will not be changed
  ChannelSet copy_mask = channels - Mask_RGB - k_spill_matte_channel_;
  row.pre_copy(row, copy_mask);
  row.copy(row, copy_mask, x, r);

  Row m_row(x, r);
  if (input(iMatte) != nullptr) { m_row.get(*input(iMatte), y, x, r, Mask_Alpha); }

  float rgb[3] = {0, 0, 0};
  afx::ReadOnlyPixel in_px(3);
  afx::Pixel out_px(3);
  for (int i = 0; i < 3; i++) {
    in_px.SetPtr(row[static_cast<Channel>(i + 1)] + x, i);  // Set const in pointers RGB. (i + 1) Chan_Red = 1
    out_px.SetPtr(row.writable(static_cast<Channel>(i + 1)) + x, i);  // Set out pointers RGB
  }
  float one = 1.0f;
  const float* m_ptr = &one;
  if (input(iMatte) != nullptr) { m_ptr = m_row[Chan_Alpha] + x; }
  for (int x0 = x; x0 < r; ++x0) {  // Loop through pixels in row
    for (int i = 0; i < 3; i++) { rgb[i] = in_px.GetVal(i); }
    float suppression_matte = 0.0f;
    float lightness_1 = 0.0f;
    float lightness_2 = 0.0f;
    float lightness_adjust = 0.0f;
    if (k_lightness_adj > 0) {
      lightness_1 = powf(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 1.0f / 3.0f);  // Lightness - cubic root of relative luminance
    }
    hue_shifter_.Rotate(rgb);  // Shift hue so mean hue is center on ideal screen hue
    float suppression = k_amount_ * (*m_ptr) * afx::SpillSuppression(rgb, k_algorithm_, color_);  // Calculate suppression
    switch (color_) {
      case afx::kGreen: {
        rgb[1] -= suppression;
        break;
      }
      case afx::kBlue: {
        rgb[2] -= suppression;
        break;
      }
    }
    hue_shifter_inv_.Rotate(rgb);
    if (k_lightness_adj > 0) {
      lightness_2 = powf(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 1.0f / 3.0f);
      lightness_adjust = k_lightness_adj * (lightness_1 / lightness_2 - 1) + 1;
      for (int i = 0; i < 3; i++) { rgb[i] = fmaxf(rgb[i] * lightness_adjust, 0.0f); }
    }
    for (int i = 0; i < 3; i++) {
      out_px[i] = rgb[i];
    }
    if (k_spill_matte_channel_) {
      suppression_matte = suppression / fmaxf(ref_suppression_, 0.0f);
    }
    foreach(z, k_spill_matte_channel_) { *(row.writable(z) + x0) = suppression_matte; }
    in_px++;
    out_px++;
    if (input(iMatte) != nullptr) { m_ptr++; }
  }
}
