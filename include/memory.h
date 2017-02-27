// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_MEMORY_H_
#define INCLUDE_MEMORY_H_

#include <jemalloc/jemalloc.h>

#include "include/bounds.h"

namespace afx {

template <typename T>
inline T* ImageMalloc(unsigned int width, unsigned int height, size_t* pitch) {
  *pitch = (((width * sizeof(T) + 63) / 64) * 64);
  return reinterpret_cast<T*>(je_aligned_alloc(64, *pitch * height));
}

inline void ImageFree(void* ptr) {
  if (ptr != nullptr) {
    je_free(ptr);
  }
  ptr = nullptr;
}

}  // namespace afx

#endif  // INCLUDE_MEMORY_H_
