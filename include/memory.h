// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef MEMORY_H_
#define MEMORY_H_

#include <jemalloc/jemalloc.h>

#include "types.h"

namespace afx {

template <typename T>
void ImageMalloc(void** ptr, size_t* pitch, const afx::Bounds& region) {
  *pitch = (((region.GetWidth() * sizeof(T) + 63) / 64) * 64);
  je_posix_memalign(ptr, 64, *pitch * region.GetHeight());
}

void ImageFree(void* ptr) {
  je_free(ptr);
}

} // namespace afx

#endif  // MEMORY_H_