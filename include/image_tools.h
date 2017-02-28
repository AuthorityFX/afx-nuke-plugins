// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_IMAGE_TOOLS_H_
#define INCLUDE_IMAGE_TOOLS_H_

#include <ipp.h>
#include <ippcore.h>
#include <ipptypes.h>
#include <ippi.h>

#include "include/image.h"
#include "include/threading.h"

namespace afx {

class Compliment {
 private:
   void ComplimentTile_(const afx::Bounds& region, afx::Image* image) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* ptr = image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *ptr = 1.0f - *ptr;
        ptr++;
      }
    }
  }

 public:
  Compliment(afx::Image* image) {
    Compute(image, image->GetBounds());
  }
  Compliment(afx::Image* image, const afx::Bounds& region) {
    Compute(image, region);
  }
  void Compute(afx::Image* image) {
    Compute(image, image->GetBounds());
  }
  void Compute(afx::Image* image, const afx::Bounds& region) {
    afx::ImageThreader threader;
    threader.ThreadImageChunks(region, boost::bind(&Compliment::ComplimentTile_, this, _1, image));
    threader.Wait();
  }
};

class Resize {
private:
  void ResizeTile_(afx::Bounds region, const afx::Image& in, afx::Image* out) {
    IppiBorderType border_style = ippBorderRepl;
    if (in.GetBounds().x1() > region.x1()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemRight); }
    if (in.GetBounds().x2() < region.x2()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemLeft); }
    if (in.GetBounds().y1() > region.y1()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemBottom); }
    if (in.GetBounds().y2() < region.y2()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemTop); }

    int spec_structure_size;
    int init_buffer_size;

    ippiResizeGetSize_32f(in.GetSize(), out->GetSize(), ippLanczos, 1, &spec_structure_size, &init_buffer_size);
    boost::scoped_ptr<Ipp8u> spec_structure_ptr(new Ipp8u(spec_structure_size));
    boost::scoped_ptr<Ipp8u> init_buffer_ptr(new Ipp8u(init_buffer_size));

    ippiResizeLanczosInit_32f(in.GetSize(), out->GetSize(), 2, reinterpret_cast<IppiResizeSpec_32f*>(spec_structure_ptr.get()), init_buffer_ptr.get());

    IppiSize dst_row_size = {out->GetBounds().GetWidth(), 1};
    int buffer_size;
    ippiResizeGetBufferSize_32f(reinterpret_cast<IppiResizeSpec_32f*>(spec_structure_ptr.get()), dst_row_size, 1, &buffer_size);
    boost::scoped_ptr<Ipp8u> buffer_ptr(new Ipp8u(buffer_size));

    IppiPoint dst_offset = {0, region.y1() - in.GetBounds().y1()};
    Ipp32f border_value = 0.0f;
    ippiResizeAntialiasing_32f_C1R(in.GetPtr(region.x1(), region.y1()), in.GetPitch(), out->GetPtr(region.x1(), region.y1()), out->GetPitch(),
                                   dst_offset, dst_row_size, border_style, &border_value, reinterpret_cast<IppiResizeSpec_32f*>(spec_structure_ptr.get()), buffer_ptr.get());
  }

public:
  Resize(const afx::Image& in, afx::Image* out) {
    afx::ImageThreader threader;
    threader.ThreadImageRows(in.GetBounds(), boost::bind(&Resize::ResizeTile_, this, _1, boost::cref(in), out));
  }
};

}  //  namespace afx

#endif  // INCLUDE_IMAGE_TOOLS_H_
