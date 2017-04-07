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

#include "include/pixel.h"
#include "include/nuke_helper.h"
#include "include/median.h"
#include "include/color_op.h"

#define ThisClass AFXDeSpill

namespace nuke = DD::Image;

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXDeSpill";
static const char* HELP = "Remove Spill";

enum inputs {
  iSource = 0,
  iMask = 1
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

class ThisClass : public nuke::Iop {
 private:
  // members to store knob values
  float k_screen_color_[3];
  float k_amount_;
  float k_lightness_adj;
  int k_algorithm_;
  nuke::ChannelSet k_spill_matte_channel_;

  afx::RotateColor hue_shifter_;
  afx::RotateColor hue_shifter_inv_;
  float ref_suppression_;
  afx::ScreenColor color_;

  void ProcessCPU(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);

 public:
  explicit ThisClass(Node* node);
  void knobs(nuke::Knob_Callback);
  int knob_changed(nuke::Knob* k);
  const char* Class() const;
  const char* node_help() const;
  static const Iop::Description d;
  Op* default_input(int input) const;
  const char* input_label(int input, char* buffer) const;

  void _validate(bool);
  void _request(int x, int y, int r, int t, nuke::ChannelMask channels, int count);
  void engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);
};
ThisClass::ThisClass(Node* node) : nuke::Iop(node) {
  // Set inputs
  inputs(2);

  // initialize knobs
  k_algorithm_ = afx::kHarder;
  k_amount_ = 1.0f;
  k_lightness_adj = 1.0f;
  k_screen_color_[0] = k_screen_color_[2] = 0.0f;
  k_screen_color_[1] = 1.0f;
}
void ThisClass::knobs(nuke::Knob_Callback f) {
  nuke::Color_knob(f, k_screen_color_, "screen_color", "Screen Color");
  nuke::Tooltip(f, "Color of Chroma Screen");
  nuke::ClearFlags(f, nuke::Knob::MAGNITUDE | nuke::Knob::SLIDER);

  nuke::Divider(f, "Settings");

  nuke::Float_knob(f, &k_amount_, "amount", "Spill Amount");
  nuke::Tooltip(f, "Amount of spill to remove");
  nuke::SetRange(f, 0.0, 1.0);
  nuke::SetFlags(f, nuke::Knob::FORCE_RANGE);

  nuke::Float_knob(f, &k_lightness_adj, "lightness_adj", "Lightness Adjustment");
  nuke::Tooltip(f, "Adjust perceived lightness");
  nuke::SetRange(f, 0.0, 1.0);
  nuke::SetFlags(f, nuke::Knob::FORCE_RANGE);

  nuke::Enumeration_knob(f, &k_algorithm_, algorithm_list, "algorithm", "Algorithm");
  nuke::Tooltip(f, "Spill Algorithm");

  nuke::Divider(f, "Output");

  nuke::ChannelSet_knob(f, &k_spill_matte_channel_, "spill_matte", "Spill Matte");
}
int ThisClass::knob_changed(nuke::Knob* k) {
  return nuke::Iop::knob_changed(k);
}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static nuke::Iop* build(Node* node) { return new ThisClass(node); }
const nuke::Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Smart Median", build);
nuke::Op* ThisClass::default_input(int input) const {
  if (input == 0) { return nuke::Iop::default_input(0); }
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

  nuke::ChannelSet out_channels = channels();
  out_channels += k_spill_matte_channel_;
  set_out_channels(out_channels);
  info_.turn_on(out_channels);

  // 1/3 hue is green, 2/3 hue is blue. center_ chosen based on max(g, b)
  // (center_ - mean) * 360 is the angular distance in degrees from ideal screen hue
  float hsv[3];
  float ref_rgb[3];
  for (int i = 0; i < 3; i++) { ref_rgb[i] = k_screen_color_[i]; }
  afx::RGBtoHSV(ref_rgb, hsv);
  float center = 0;
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
void ThisClass::_request(int x, int y, int r, int t, nuke::ChannelMask channels, int count) {
  nuke::ChannelSet req_channels = channels;
  req_channels += nuke::Mask_RGB;
  input(iSource)->request(x, y, r, t, req_channels, count);  // Only request RGB
  if (input(iMask) != nullptr) { input(iMask)->request(x, y, r, t, nuke::Mask_Alpha, count); }
}
void ThisClass::engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  callCloseAfter(0);
  ProcessCPU(y, x, r, channels, row);
}
void ThisClass::ProcessCPU(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  nuke::ChannelSet req_channels = channels;
  req_channels += nuke::Mask_RGB;  // Add RGB to channels
  row.get(input0(), y, x, r, req_channels);  // Request all channels
  // Copy channels that will not be changed
  nuke::ChannelSet copy_mask = channels - nuke::Mask_RGB - k_spill_matte_channel_;
  row.pre_copy(row, copy_mask);
  row.copy(row, copy_mask, x, r);

  nuke::Row m_row(x, r);
  if (input(iMask) != nullptr) { m_row.get(*input(iMask), y, x, r, nuke::Mask_Alpha); }

  float rgb[3] = {0, 0, 0};
  afx::Pixel<const float> in_px(3);
  afx::Pixel<float> out_px(3);
  for (int i = 0; i < 3; i++) {
    in_px.SetPtr(row[static_cast<nuke::Channel>(i + 1)] + x, i);  // Set const in pointers RGB. (i + 1) Chan_Red = 1
    out_px.SetPtr(row.writable(static_cast<nuke::Channel>(i + 1)) + x, i);  // Set out pointers RGB
  }
  float one = 1.0f;
  const float* m_ptr = &one;
  if (input(iMask) != nullptr) { m_ptr = m_row[nuke::Chan_Alpha] + x; }
  for (int x0 = x; x0 < r; ++x0) {  // Loop through pixels in row
    for (int i = 0; i < 3; i++) { rgb[i] = in_px.GetVal(i); }
    float suppression_matte = 0.0f;
    float lightness_1 = 0.0f;
    float lightness_2 = 0.0f;
    float lightness_adjust = 0.0f;
    if (k_lightness_adj > 0) {
      lightness_1 = powf(0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2], 1.0f / 3.0f);  // Lightness - cubic root of relative luminance
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
      lightness_2 = powf(0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2], 1.0f / 3.0f);
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
    if (input(iMask) != nullptr) { m_ptr++; }
  }
}
