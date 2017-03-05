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
#include <DDImage/Tile.h>
#include <DDImage/Thread.h>
#include <DDImage/NukeWrapper.h>

#include <math.h>

#include <cuda_runtime.h>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <algorithm>

#include "include/threading.h"
#include "include/image.h"
#include "include/pixel.h"
#include "include/cuda_helper.h"
#include "include/nuke_helper.h"
#include "include/glow.h"
#include "include/color_op.h"

#define ThisClass AFXGlow

namespace nuke = DD::Image;

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXGlow";
static const char* HELP = "Glow";

enum inputs {
  iSource = 0,
  iMask = 1
};

class ThisClass : public nuke::Iop {
 private:
  // members to store knob values
  float k_exposure_;
  float k_threshold_;
  float k_threshold_falloff_;
  float k_size_;
  float k_softness_;
  float k_diffusion_;
  float k_quality_;
  bool k_replicate_;
  float k_replicate_falloff_;
  bool k_expand_box_;

  // members to store processed knob values
  float exposure_, threshold_, threshold_falloff_, size_, softness_, diffusion_, quality_;
  bool replicate_, expand_box_;
  float replicate_falloff_;

  afx::Glow glow_;

  boost::mutex mutex_;
  bool first_time_GPU_;
  bool first_time_CPU_;
  nuke::Lock lock_;

  afx::Bounds req_bnds_, format_bnds_, format_f_bnds_;
  float proxy_scale_;

  afx::CudaProcess cuda_process_;
  afx::CudaImageArray row_c_imgs_;
  afx::CudaImageArray in_c_imgs_;
  afx::CudaStreamArray streams_;

  afx::ImageArray in_imgs_;
  afx::ImageArray out_imgs_;

  afx::ImageThreader threader_;

  void ProcessCUDA(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);
  void ProcessCPU(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row);

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
ThisClass::ThisClass(Node* node) : Iop(node) {
  // Set inputs
  inputs(2);

  first_time_GPU_ = true;
  first_time_CPU_ = true;

  // initialize knobs
  k_exposure_ = 0.0f;
  k_threshold_ = 1.0f;
  k_threshold_falloff_ = 0.25f;
  k_size_ = 250.0f;
  k_softness_ = 0.0;
  k_diffusion_ = 0.5f;
  k_quality_ = 0.5f;
  k_replicate_ = false;
  k_replicate_falloff_ = 0.5;
  k_expand_box_ = false;
}
void ThisClass::knobs(nuke::Knob_Callback f) {
  nuke::Float_knob(f, &k_exposure_, "exposure", "Exposure");
  nuke::Tooltip(f, "Exposure");
  nuke::SetRange(f, -5.0, 5.0);

  nuke::Float_knob(f, &k_threshold_, "threshold", "Threshold");
  nuke::Tooltip(f, "Threshold");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Float_knob(f, &k_threshold_falloff_, "threshold_falloff", "Threshold Falloff");
  nuke::Tooltip(f, "Threshold falloff");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Float_knob(f, &k_size_, "size", "Glow Size");
  nuke::Tooltip(f, "Glow size in pixels");
  nuke::SetRange(f, 1.0, 500.0);

  nuke::Float_knob(f, &k_softness_, "softness", "Softness");
  nuke::Tooltip(f, "Softness");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Float_knob(f, &k_diffusion_, "Diffusion", "Diffusion");
  nuke::Tooltip(f, "Diffusion");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Float_knob(f, &k_quality_, "quality", "Quality");
  nuke::Tooltip(f, "Quality multiplier");
  nuke::SetRange(f, 0.0, 1.0);

  nuke::Bool_knob(f, &k_replicate_, "replicate", "Replicate Border");
  nuke::Tooltip(f, "Replicate borders");
  nuke::SetFlags(f, nuke::Knob::STARTLINE);

  nuke::Float_knob(f, &k_replicate_falloff_, "replicate_falloff", "Replicate Falloff");
  nuke::Tooltip(f, "Replicate Falloff");
  nuke::ClearFlags(f, nuke::Knob::STARTLINE);
  nuke::SetRange(f, 0, 1);

  nuke::Bool_knob(f, &k_expand_box_, "expand_box", "Expand Box");
  nuke::Tooltip(f, "Expand box");
  nuke::SetFlags(f, nuke::Knob::STARTLINE);
}
const char* ThisClass::Class() const { return CLASS; }
const char* ThisClass::node_help() const { return HELP; }
static nuke::Iop* build(Node* node) { return (new nuke::NukeWrapper(new ThisClass(node)))->channelsRGBoptionalAlpha(); }
const nuke::Op::Description ThisClass::d(CLASS, "AuthorityFX/AFX Glow", build);
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
      input_longlabel(input) = "Glow Matte";
      return "Glow Matte";
    }
    default: {
      return 0;
    }
  }
}
void ThisClass::_validate(bool) {
  copy_info(0);

  format_bnds_ = afx::BoxToBounds(input(0)->format());
  format_f_bnds_ = afx::BoxToBounds(input(0)->full_size_format());
  proxy_scale_ = static_cast<float>(format_bnds_.GetWidth()) / static_cast<float>(format_f_bnds_.GetWidth());

  exposure_ = k_exposure_;
  threshold_ = k_threshold_;
  threshold_falloff_ = fminf(fmaxf(10.0f * k_threshold_falloff_, 1.0f), 125.0f);
  size_ = proxy_scale_ * k_size_;
  softness_ = proxy_scale_ * afx::AfxClamp<float>(k_softness_, 0.0, 10.0f);
  diffusion_ = afx::AfxClamp<float>(k_diffusion_, 0.0, 1.0f);
  quality_ = afx::AfxClamp<float>(k_quality_, 0.0f, 1.0f);
  replicate_ = k_replicate_;
  replicate_falloff_ = k_replicate_falloff_;
  expand_box_ = k_expand_box_;

  glow_.CreateKernel(size_, softness_, diffusion_, quality_);

  if (k_expand_box_) { info_.pad(glow_.RequiredPadding()); }

  info_.black_outside();
}
void ThisClass::_request(int x, int y, int r, int t, nuke::ChannelMask channels, int count) {
  unsigned int pad = glow_.RequiredPadding();
  nuke::Box req_box(x + pad, y + pad, r + pad, t + pad);
  input0().request(req_box, channels, count);
  if (input(iMask) != nullptr) { input(iMask)->request(req_box, nuke::Mask_Alpha, count); }

  req_bnds_.SetBounds(x, y, r - 1, t - 1);
}
void ThisClass::_open() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  in_imgs_.Clear();
  out_imgs_.Clear();
}
void ThisClass::_close() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  in_imgs_.Clear();
  out_imgs_.Clear();
}
void ThisClass::engine(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  callCloseAfter(0);
  ProcessCPU(y, x, r, channels, row);
}
void ThisClass::ProcessCUDA(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);
}
void ThisClass::ProcessCPU(int y, int x, int r, nuke::ChannelMask channels, nuke::Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);

  {
    nuke::Guard guard(lock_);
    if (first_time_CPU_) {
      first_time_CPU_ = false;
      afx::Bounds glow_bnds = req_bnds_.GetPadBounds(glow_.RequiredPadding());
      afx::Bounds plane_bnds = glow_bnds;
      plane_bnds.Intersect(afx::InputBounds(input(0)));

      if (aborted()) { return; }

      afx::Image mask_image;
      if (input(iMask) != nullptr ) {
        nuke::ImagePlane mask_plane;
        input(iMask)->fetchPlane(mask_plane);
        mask_image.Allocate(plane_bnds);
        mask_image.MemCpyIn(mask_plane.readable(), afx::GetPlanePitch(mask_plane));
      }

      // Used as input to convolution for all channels
      afx::Image glow_source(glow_bnds);

      nuke::ChannelSet done;
      foreach(z, channels) {
        if (!(done & z) && colourIndex(z) < 3) {  // Handle color channels
          bool has_all_rgb = true;
          nuke::Channel rgb_chan[3];
          for (int i = 0; i < 3; ++i) {
            rgb_chan[i] = brother(z, i);  // Find brother rgb channel
            if (rgb_chan[i] == nuke::Chan_Black || !(channels & rgb_chan[i])) { has_all_rgb = false; }  // If brother does not exist
          }
          if (has_all_rgb) {
            // Create image layer
            afx::ImageLayer image_layer;
            for (int i = 0; i < 3; ++i) {
              done += rgb_chan[i];  // Add channel to done channel set
              image_layer.AddImage(plane_bnds);
              nuke::ImagePlane channel_plane;
              input(iSource)->fetchPlane(channel_plane);
              image_layer[i]->MemCpyIn(channel_plane.readable(), afx::GetPlanePitch(channel_plane));
            }

            afx::Threshold thresholder;
            thresholder.ThresholdLayer(&image_layer, mask_image, threshold_, threshold_falloff_);

            for (int i = 0; i < 3; ++i) {
              afx::Image* image_ptr = image_layer[i];

              // Add output image for channel
              out_imgs_.Add(req_bnds_);
              afx::Image* out_ptr = out_imgs_.GetBackPtr();
              out_ptr->AddAttribute("channel", rgb_chan[i]);

              // Extend boarders from plane_bnds to glow_bnds
              afx::BorderExtender border_extender;
              if (replicate_) {
                border_extender.RepeatFalloff(*image_ptr, &glow_source, replicate_falloff_);
              } else {
                border_extender.Constant(*image_ptr, &glow_source, 0.0f);
              }
              // Do glow
              glow_.DoGlow(glow_source, out_ptr);
            }
          }
        }
        if (!(done & z)) {  // Handle non color channel
          done += z;

          // Copy planes to afx::Images
          // Threshold
          // Extend Borders
          // do Glow

        }
      }

    }
  }  // End first time guard

  if (aborted()) { return; }

  foreach(z, channels) {
    out_imgs_.GetPtrByAttribute("channel", z)->MemCpyOut(row.writable(z) + x, row_bnds.GetWidth() * sizeof(float), row_bnds);
  }
}
