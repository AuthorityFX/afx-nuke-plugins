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
  iMatte = 1
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
  int k_replicate_depth_;
  bool k_expand_box_;

  // members to store processed knob values
  float exposure_, threshold_, threshold_falloff_, size_, softness_, diffusion_, quality_;
  bool replicate_, expand_box_;
  int replicate_depth_;

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

  void GetInputRGB(const afx::Bounds& region, const nuke::ImagePlane& in_plane, const nuke::ImagePlane& matte_plane, const nuke::Channel (&chan_ref)[3]);
  void GetInput(const afx::Bounds& region, const nuke::ImagePlane& in_plane, const nuke::ImagePlane& matte_plane, nuke::Channel z);

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
  k_replicate_depth_ = 5;
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

  nuke::Int_knob(f, &k_replicate_depth_, "replicate_depth", "Replicate Depth");
  nuke::Tooltip(f, "Replicate depth");
  nuke::SetRange(f, 0, 25);

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
  replicate_depth_ = static_cast<int>(proxy_scale_ * std::max(k_replicate_depth_, 0));
  expand_box_ = k_expand_box_;

  glow_.ComputeGaussSize(size_, softness_, diffusion_, quality_);

  if (k_expand_box_) { info_.pad(glow_.GetKernelPadding()); }

  info_.black_outside();
}
void ThisClass::_request(int x, int y, int r, int t, nuke::ChannelMask channels, int count) {
  unsigned int pad = glow_.GetKernelPadding();
  nuke::Box req_box(x + pad, y + pad, r + pad, t + pad);
  input0().request(req_box, channels, count);
  if (input(iMatte) != nullptr) { input(iMatte)->request(req_box, nuke::Mask_Alpha, count); }

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

      glow_.InitKernel(exposure_, &threader_);

      afx::Bounds glow_bnds = req_bnds_.GetPadBounds(glow_.GetKernelPadding());
      afx::Bounds plane_bnds = glow_bnds;
      plane_bnds.Intersect(afx::InputBounds(input(0)));

      nuke::ImagePlane in_plane(afx::BoundsToBox(plane_bnds), false, channels);  // Create plane "false" = non-packed.
      nuke::ImagePlane matte_plane(afx::BoundsToBox(plane_bnds), false, nuke::Mask_Alpha);  // Create plane "false" = non-packed.
      if (aborted()) {
        threader_.Synchonize();
        return;
      }  // Return if render aborted
      input0().fetchPlane(in_plane);  // Fetch plane
      if (input(iMatte) != nullptr ) { input(iMatte)->fetchPlane(matte_plane); }  // Fetch matte plane

      nuke::ChannelSet done;
      foreach(z, in_plane.channels()) {
        if (!(done & z) && colourIndex(z) < 3) {  // Handle color channels
          bool has_all_rgb = true;
          nuke::Channel rgb_chan[3];
          for (int i = 0; i < 3; ++i) {
            rgb_chan[i] = brother(z, i);  // Find brother rgb channel
            if (rgb_chan[i] == nuke::Chan_Black || !(channels & rgb_chan[i])) { has_all_rgb = false; }  // If brother does not exist
          }
          if (has_all_rgb) {
            for (int i = 0; i < 3; ++i) {
              done += rgb_chan[i];  // Add channel to done channel set
              in_imgs_.Add(glow_bnds);
              in_imgs_.GetBackPtr()->AddAttribute("channel", rgb_chan[i]);
            }
            threader_.ThreadImageChunks(glow_bnds, boost::bind(&ThisClass::GetInputRGB, this, _1, boost::cref(in_plane), boost::cref(matte_plane), boost::cref(rgb_chan)));
            for (int i = 0; i < 3; ++i) {
              out_imgs_.Add(req_bnds_);
              out_imgs_.GetBackPtr()->AddAttribute("channel", rgb_chan[i]);
            }
          }
        }
        if (!(done & z)) {  // Handle non color channel
          done += z;
          in_imgs_.Add(glow_bnds);
          in_imgs_.GetBackPtr()->AddAttribute("channel", z);
          threader_.ThreadImageChunks(glow_bnds, boost::bind(&ThisClass::GetInput, this, _1, boost::cref(in_plane), boost::cref(matte_plane), z));
          out_imgs_.Add(req_bnds_);
          out_imgs_.GetBackPtr()->AddAttribute("channel", z);
        }
      }

      if (aborted()) {
        threader_.Synchonize();
        return;
      }  // Return if render aborted

      threader_.Synchonize();
      for (afx::ImageArray::ptr_list_it it = in_imgs_.GetBegin(); it != in_imgs_.GetEnd(); ++it) {
        afx::Image* in_ptr = &(*it);
        afx::Image* out_ptr = out_imgs_.GetPtrByAttribute("channel", in_ptr->GetAttribute("channel"));
        threader_.AddWork(boost::bind(&afx::Glow::Convolve, &glow_, boost::cref(*in_ptr), out_ptr));
      }
      threader_.Synchonize();
    }
  }  // End first time guard

  if (aborted()) { return; }

  foreach(z, channels) {
    out_imgs_.GetPtrByAttribute("channel", z)->MemCpyOut(row.writable(z) + x, row_bnds.GetWidth() * sizeof(float), row_bnds);
  }
}

void ThisClass::GetInputRGB(const afx::Bounds& region, const nuke::ImagePlane& in_plane, const nuke::ImagePlane& matte_plane, const nuke::Channel (&rgb_chan)[3]) {
  afx::Bounds plane_bnds = afx::BoxToBounds(in_plane.bounds());
  afx::Pixel<const float> in(3);
  afx::Pixel<float> out(3);
  const float one = 1.0f;
  const float* m_ptr = &one;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    for (int i = 0; i < 3; ++i) {
      in.SetPtr(&in_plane.readable()[in_plane.chanNo(rgb_chan[i]) * in_plane.chanStride() + in_plane.rowStride() * (plane_bnds.ClampY(y) - plane_bnds.y1()) +
                plane_bnds.ClampX(region.x1()) - plane_bnds.x1()], i);
      out.SetPtr(in_imgs_.GetPtrByAttribute("channel", rgb_chan[i])->GetPtr(region.x1(), y), i);
    }
    if (input(iMatte) != nullptr) { m_ptr =
                                            &matte_plane.readable()[matte_plane.chanNo(nuke::Chan_Alpha) * matte_plane.chanStride() + matte_plane.rowStride() *
                                            (plane_bnds.ClampY(y) - plane_bnds.y1()) + plane_bnds.ClampX(region.x1()) - plane_bnds.x1()];
                                  }
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float luma = 0.3f * in.GetVal(0) + 0.59f * in.GetVal(1) + 0.11f * in.GetVal(2);
      if (luma < threshold_) {
        float scale = powf((luma / threshold_), threshold_falloff_);
        for (int i = 0; i < 3; ++i) {
          out.SetVal(fmaxf(in.GetVal(i) * scale * *m_ptr, 0.0f), i);
        }
      } else {
        for (int i = 0; i < 3; ++i) { out.SetVal(fmaxf(in.GetVal(i) * *m_ptr, 0.0f), i); }
      }
      if (!plane_bnds.CheckBounds(x, y)) {
        if (!replicate_ || x < plane_bnds.x1() - replicate_depth_ || x > plane_bnds.x2() + replicate_depth_ || y < plane_bnds.y1() - replicate_depth_ ||
            y > plane_bnds.y2() + replicate_depth_) {
          for (int i = 0; i < 3; ++i) { out.SetVal(0.0f, i); }
        }
      }
      out++;
      if (x >= plane_bnds.x1() && x < plane_bnds.x2()) {
        in++;
        if (input(iMatte) != nullptr) { m_ptr++; }
      }
    }
  }
}

void ThisClass::GetInput(const afx::Bounds& region, const nuke::ImagePlane& in_plane, const nuke::ImagePlane& matte_plane, nuke::Channel z) {
  afx::Bounds plane_bnds = afx::BoxToBounds(in_plane.bounds());
  const float one = 1.0f;
  const float* m_ptr = &one;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    const float* in_ptr = &in_plane.readable()[in_plane.chanNo(z) * in_plane.chanStride() + in_plane.rowStride() * (plane_bnds.ClampY(y) - plane_bnds.y1()) +
                          plane_bnds.ClampX(region.x1()) - plane_bnds.x1()];
    if (input(iMatte) != nullptr) { m_ptr =
                                            &matte_plane.readable()[matte_plane.chanNo(nuke::Chan_Alpha) * matte_plane.chanStride() + matte_plane.rowStride() *
                                            (plane_bnds.ClampY(y) - plane_bnds.y1()) + plane_bnds.ClampX(region.x1()) - plane_bnds.x1()];
                                  }
    float* out_ptr = in_imgs_.GetPtrByAttribute("channel", z)->GetPtr(region.x1(), y);
    for (int x = region.x1(); x <= region.x2(); ++x) {
      if (*in_ptr < threshold_) {
        *out_ptr = fmaxf(*in_ptr * powf((*in_ptr / threshold_), threshold_falloff_) * *m_ptr, 0.0f);
      } else {
        *out_ptr = fmaxf(*in_ptr * *m_ptr, 0.0f);
      }
      if (!plane_bnds.CheckBounds(x, y)) {
        if (!replicate_ || x < plane_bnds.x1() - replicate_depth_ || x > plane_bnds.x2() + replicate_depth_ || y < plane_bnds.y1() - replicate_depth_ ||
            y > plane_bnds.y2() + replicate_depth_) {
          *out_ptr = 0.0f;
        }
      }
      out_ptr++;
      if (plane_bnds.CheckBoundsX(x)) {
        in_ptr++;
        if (input(iMatte) != nullptr) { m_ptr++; }
      }
    }
  }
}
