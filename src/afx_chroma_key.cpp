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

#include <cuda_runtime.h>
#include <math.h>

#include "threading.h"
#include "cuda_helper.h"
#include "median.h"
#include "color_op.h"

// The class name must match exactly what is in the meny.py: nuke.createNode(CLASS)
static const char* CLASS = "AFXChromaKey";
static const char* HELP = "Chroma Keyer";

#define ThisClass AFXChromaKey

extern "C" void MetricsCuda(cudaTextureObject_t r_tex, cudaTextureObject_t g_tex, cudaTextureObject_t b_tex, cudaTextureObject_t m_tex, afx::Bounds in_bnds);
extern "C" void CoreMatteCuda(afx::Bounds in_bnds);

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

  float k_screen_mean_[3];

  float mean_rgb_[3];
  float mean_lab_[3];
  float std_dev_lab_[3];

  // members to store processed knob values
  float smoothness_;

  boost::mutex mutex_;
  bool first_time_GPU_;
  bool first_time_CPU_;
  Lock lock_;

  afx::Bounds req_bnds_, format_bnds, format_f_bnds_;
  float proxy_scale_;

  afx::CudaProcess cuda_process_;
  afx::CudaImageArray row_imgs_;
  afx::CudaImageArray in_imgs_;
  afx::CudaStreamArray streams_;

  afx::Threader threader_;

  int maximum_inputs() const { return 2; }
  int minimum_inputs() const { return 2; }

  void MetricsCPU(const afx::Bounds& region, const ImagePlane& source, const ImagePlane& matte, float* ref_hsv, double* sum_rgb, double* sum, double* sum_sqrs, unsigned int& num);
  void ProcessCUDA(int y, int x, int r, ChannelMask channels, Row& row);
  void ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row);

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

  first_time_GPU_ = true;
  first_time_CPU_ = true;

  try {
    cuda_process_.MakeReady();
  } catch (cudaError err) {}

  //initialize knobs
  k_screen_color_[0] = k_screen_color_[2] = 0;
  k_screen_color_[1] = 1;
  k_specify_source_frame_ = false;
  k_source_frame_ = 1.0f;

  k_chroma_clean_ = 0.5f;
  k_luma_clean_ = 0.5f;
  k_falloff_ = 0.5f;
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
  first_time_GPU_ = true;
  first_time_CPU_ = true;
}
void ThisClass::_close() {
  first_time_GPU_ = true;
  first_time_CPU_ = true;
  in_imgs_.Clear();
  row_imgs_.Clear();
  streams_.Clear();
}
void ThisClass::engine(int y, int x, int r, ChannelMask channels, Row& row) {
  callCloseAfter(0);
  try {
    throw cudaErrorIllegalAddress;
    ProcessCUDA(y, x, r, channels, row);
  } catch (cudaError err) {
    ProcessCPU(y, x, r, channels, row);
  }
}
void ThisClass::ProcessCUDA(int y, int x, int r, ChannelMask channels, Row& row) {
  afx::Bounds row_bnds(x, y, r - 1, y);

  if (first_time_GPU_) {
    Guard guard(lock_);
    if (first_time_GPU_) {
      cuda_process_.CheckReady();
      in_imgs_.Clear();
      row_imgs_.Clear();
      streams_.Clear();
      ImagePlane source_plane(Box(req_bnds_.x1(), req_bnds_.y1(), req_bnds_.x2() + 1, req_bnds_.y2() + 1), false, Mask_RGB); // Create plane "false" = non-packed.
      ImagePlane screen_plane(Box(req_bnds_.x1(), req_bnds_.y1(), req_bnds_.x2() + 1, req_bnds_.y2() + 1), false, Mask_RGB); // Create plane "false" = non-packed.
      if (k_specify_source_frame_) {
        input(iSource)->fetchPlane(source_plane);
        if (input(iScreenMatte) != nullptr) { input(iScreenMatteAtTime)->fetchPlane(screen_plane); }
      } else {
        input(iSourceAtTime)->fetchPlane(source_plane);
        if (input(iScreenMatte) != nullptr) { input(iScreenMatteAtTime)->fetchPlane(screen_plane); }
      }
      afx::CudaStreamArray streams;
      afx::CudaImage* rgb[3];
      foreach (z, info_.channels()) { // For each channel in plane
        if (z == Chan_Red || z == Chan_Green || z == Chan_Blue) {
          in_imgs_.AddImage(req_bnds_); // Create cuda image for plane
          in_imgs_.GetBackPtr()->AddAttribute("channel", z); // Add channel attribute to image
          streams.Add(); // Add stream
          in_imgs_.GetBackPtr()->MemCpyToDevice(&source_plane.readable()[source_plane.chanNo(z) * source_plane.chanStride()], source_plane.rowStride() * sizeof(float),
                                                streams.GetBackPtr()->GetStream()); // Copy mem host plane to cuda on stream.
          switch (z) {
            case Chan_Red: {
              rgb[0] = in_imgs_.GetBackPtr();
            }
            case Chan_Green: {
              rgb[1] = in_imgs_.GetBackPtr();
            }
            case Chan_Blue: {
              rgb[2] = in_imgs_.GetBackPtr();
            }
          }
        }
      }
      afx::CudaImage* screen_matte_ptr = nullptr;
      if (input(iScreenMatte) != nullptr) {
        in_imgs_.AddImage(req_bnds_); // Create cuda image for plane
        in_imgs_.GetBackPtr()->AddAttribute("matte", iScreenMatte); // Add matte input number attribute to image
        streams.Add();
        in_imgs_.GetBackPtr()->MemCpyToDevice(&screen_plane.readable()[screen_plane.chanNo(Chan_Alpha) * screen_plane.chanStride()],
                                              screen_plane.rowStride() * sizeof(float), streams.GetBackPtr()->GetStream()); // Copy mem host plane to cuda on stream.
        screen_matte_ptr = in_imgs_.GetBackPtr();
      }
      cuda_process_.Synchonize(); // Sync all streams
      in_imgs_.CreateTextures(); // Create texture objects for all cuda images
      streams.Clear(); // Delete streams
      //MetricsCuda(rgb[0]->GetTexture(), rgb[1]->GetTexture(), rgb[2]->GetTexture(), screen_matte_ptr != nullptr ? screen_matte_ptr->GetTexture() : 0,  req_bnds_);
      first_time_GPU_ = false;
    }
  } // End first time guard
}
void ThisClass::ProcessCPU(int y, int x, int r, ChannelMask channels, Row& row) {

  if (first_time_CPU_) {
    Guard guard(lock_);
    if (first_time_CPU_) {
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

      float ref_hsv[3];
      afx::RGBtoHSV(k_screen_color_, ref_hsv);

      unsigned int num = 0;
      double sum_rgb[3] = {0, 0, 0};
      double sum[3] = {0, 0, 0};
      double sum_sqrs[3] = {0, 0, 0};

      threader_.ThreadImageChunks(format_bnds, boost::bind(&ThisClass::MetricsCPU, this, _1, boost::ref(source_plane), boost::ref(matte_plane), ref_hsv, sum_rgb, sum, sum_sqrs, boost::ref(num)));
      threader_.Synchonize();

      //Calculate mean and stdDev
      num = std::max(num, (unsigned int)1);
      for (int i = 0; i < 3; ++i) {
        mean_rgb_[i] = (float)(sum_rgb[i] / (double)num);
        mean_lab_[i] = (float)(sum[i] / (double)num);
        std_dev_lab_[i] = (float)sqrt((sum_sqrs[i] - pow(sum[i], 2.0) / (double)num) / (double)num);
      }
      first_time_CPU_ = false;
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
    afx::RGBtoLab(lab, lab);
    afx::RGBtoHSV(hsv, hsv);
    rad = powf(fabsf(lab[1] - mean_lab_[1]), 2.0f) / powf(3.0f * k_chroma_clean_ * std_dev_lab_[1], 2.0f)
        + powf(fabsf(lab[2] - mean_lab_[2]), 2.0f) / powf(3.0f * k_chroma_clean_ * std_dev_lab_[2], 2.0f)
        + powf(fabsf(lab[0] - mean_lab_[0]), 2.0f) / powf(3.0f * k_luma_clean_   * std_dev_lab_[0], 2.0f);
    s_res = v_res = 0.0f;
    rad = afx::max3(rad, s_res, v_res);
    rad = fmaxf(rad - 1.0f, 0.0f);
    matte = 0.0f;
    if (rad <= smoothness_) {
      matte = rad / smoothness_;
    } else {
      matte = 1.0f;
    }
    for (int i = 0; i < 3; ++i) {
      out.SetVal(in.GetVal(i) * matte, i);
    }
    out.SetVal(matte, 3);
    in++;
    out++;
  }
}

// TODO I want to completely change this algorithm
// Currently this is what I'm doing:
// Find mean and standard deviation.
// Create 3D elipsoid or radius 3 * std_dev centered at mean.
// Plug in abs(pixel - mean) for x, y ,z in elipsoid equation
// If value is greater than 1 or less, pixel is chroma screen.
//
// The mean and std_dev can be skewed by the size of the mask areas. A big area will carry more weight.
// I want to sample only the points within the mask region and create a non-convex hull boundind the sample points
// Furthermore, I plotted the A B of LAB space. It's not a normal distribution and it doesn't look like an elipse.
//
// This is what I'd like to do
//
// http://www.geosensor.net/phpws/index.php?module=pagemaster&PAGE_user_op=view_page&PAGE_id=13
//
// Another non-convex hull can be constructed to represent the foreground.
// The two hulls can be booleaned
// To check if a pixel is part of the chroma screen
// First check bounding region. http://geomalgorithms.com/a08-_containers.html
// If within bounding region, check using winding number  http://geomalgorithms.com/a03-_inclusion.html
//
// To blend between screen and forground, if point is not within screen non-convex hull
// find distance to nearest point non-convex hull point.
//
// Convex hull points should be stored in a KD Tree to allow for fast nearest neighbor search.
//
// http://www.geosensor.net/papers/duckham08.PR.pdf
// non-convex hull algorithm

// https://en.wikipedia.org/wiki/Delaunay_triangulation
// Delaunay_triangulation


void ThisClass::MetricsCPU(const afx::Bounds& region, const ImagePlane& source, const ImagePlane& matte, float* ref_hsv, double* sum_rgb,
                           double* sum, double* sum_sqrs, unsigned int& num) {

  afx::Bounds plane_bnds(source.bounds().x(), source.bounds().y(), source.bounds().r() - 1, source.bounds().t() - 1);

  float rgb[3];
  float hsv[3];

  unsigned int l_num = 0;
  double l_sum_rgb[3] = {0, 0, 0};
  double l_sum[3] = {0, 0, 0};
  double l_sumSqrs[3] = {0, 0, 0};

  afx::ReadOnlyPixel in(4);
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

    for (int x = region.x1(); x <= region.x2(); x+=4) { // TODO every 4th row for performance
      //Convert HSV and convert pixel to HSV
      for (int i = 0; i < 3; i++) { rgb[i] = in.GetVal(i); }
      afx::RGBtoHSV(rgb, hsv);
      if (*m_ptr > 0.5) { //Compute stats from screen
        if (fabsf(hsv[0] - ref_hsv[0]) < 0.1f ) {
          l_num++;
          float lab[3];
          afx::RGBtoLab(rgb, lab);
          for (int i = 0; i < 3; i++) {
            l_sum_rgb[i] += rgb[i];
            l_sum[i] += lab[i];
            l_sumSqrs[i] += lab[i] * lab[i];
          }
        }
      }
      in.NextPixel();
      in.NextPixel();
      in.NextPixel();
      in.NextPixel();
      if (input(iScreenMatte) != nullptr) {
        m_ptr++;
        m_ptr++;
        m_ptr++;
        m_ptr++;
      }
    }
  }
  boost::mutex::scoped_lock lock(mutex_);
  num += l_num;
  for (int i = 0; i < 3; ++i) {
    sum_rgb[i] += l_sum_rgb[i];
    sum[i] += l_sum[i];
    sum_sqrs[i] += l_sumSqrs[i];
  }
}
