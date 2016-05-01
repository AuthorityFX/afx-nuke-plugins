// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "types.h"

__global__
void KernelMedian(cudaTextureObject_t tex, float* row_ptr, long int row_pitch, int med_size_i, int med_size_o, int med_n_i, int med_n_o, float m_lerp, float m_i_lerp, float sharpness, afx::Bounds img_bnds, afx::Bounds row_bnds, int med_size_2) {

  // x and y in where 0 is the bottom left pixel
  int x_cuda =  blockIdx.x * blockDim.x + threadIdx.x;
  int y_cuda =  blockIdx.y * blockDim.y + threadIdx.y;
  if (x_cuda > row_bnds.GetWidth() - 1) { return; }
  if (y_cuda > row_bnds.GetHeight() - 1) { return; }
  // Add row lower bound to shift for start of row. Subtract img lower bound to move to 0,0
  int x = x_cuda + row_bnds.x1() - img_bnds.x1();
  int y = y_cuda + row_bnds.y1() - img_bnds.y1();

  float sum_i = 0.0f, sum_o = 0.0f, sum_sq_i = 0.0f, sum_sq_o = 0.0f;
  float median = 0.0f, median_i = 0.0f, median_o = 0.0f, mean = 0.0f, std_dev = 0.0f;

  //http://ndevilla.free.fr/median/median/src/torben.c
  // Initialize min and max to input pixel value
  float min_i, max_i;
  float min_o, max_o;
  int less, greater, equal;
  float guess, maxltguess, mingtguess;
  min_i = max_i = min_o = max_o = tex2D<float>(tex, x, y);

  int y_i = 0; // Counter to find border pixels of median window
  for (int y0 = y - med_size_o; y0 <= y + med_size_o; ++y0) {
    int x_i = 0; // Counter to find border pixels of median window
    for (int x0 = x - med_size_o; x0 <= x + med_size_o; ++x0) {
      float px = tex2D<float>(tex, x0, y0);
      // If on outside ring of pixels
      if (y_i == 0 || y_i == med_size_2 || x_i == 0 || x_i == med_size_2) {
        sum_o += px;
        sum_sq_o += px * px;
      } else { // If on inside ring of pixels
        sum_i += px;
        sum_sq_i += px * px;
        if (px < min_i) { min_i = px; }
        if (px > max_i) { max_i = px; }
      }
      if (px < min_o) { min_o = px; }
      if (px > max_o) { max_o = px; }
      x_i++;
    }
    y_i++;
  }

  //http://ndevilla.free.fr/median/median/src/torben.c
  // Find median of inside only if not integer median size
  int half_i = (med_n_i + 1) / 2;
  while (m_lerp > 0) {
    less = 0; greater = 0; equal = 0;
    guess = (min_i + max_i) / 2;
    maxltguess = min_i;
    mingtguess = max_i;
    for (int y0 = y - med_size_i; y0 <= y + med_size_i; ++y0) {
      for (int x0 = x - med_size_i; x0 <= x + med_size_i; ++x0) {
        float px = tex2D<float>(tex, x0, y0);
        if (px < guess) {
          less++;
          if (px > maxltguess) { maxltguess = px; }
        } else if (px > guess) {
          greater++;
          if (px < mingtguess) { mingtguess = px; }
        } else { equal++; }
      }
    }
    if (less <= half_i && greater <= half_i) {
      break;
    } else if (less > greater) {
      max_i = maxltguess;
    } else {
      min_i = mingtguess;
    }
  } // end of while loop
  if (m_lerp > 0) {
    if (less >= half_i) {
      median_i = maxltguess;
    } else if (less + equal >= half_i) {
      median_i = guess;
    } else {
      median_i = mingtguess;
    }
  } else {
    median_i = min_i;
  }

  // Find median of outside
  int half_o = ( med_n_o + 1) / 2;
  while (1) {
    less = 0; greater = 0; equal = 0;
    guess = (min_o + max_o) / 2;
    maxltguess = min_o;
    mingtguess = max_o;
    for (int y0 = y - med_size_o; y0 <= y + med_size_o; ++y0) {
      for (int x0 = x - med_size_o; x0 <= x + med_size_o; ++x0) {
        float px = tex2D<float>(tex, x0, y0);
        if (px < guess) {
          less++;
          if (px > maxltguess) { maxltguess = px; }
        } else if (px > guess) {
          greater++;
          if (px < mingtguess) { mingtguess = px; }
        } else { equal++; }
      }
    }
    if (less <= half_o && greater <= half_o) {
      break; // Break while loop
    } else if (less > greater) {
      max_o = maxltguess;
    } else {
      min_o = mingtguess;
    }
  } // end of while loop
  if (less >= half_o) {
    median_o = maxltguess;
  } else if (less + equal >= half_o) {
    median_o = guess;
  } else {
    median_o = mingtguess;
  }

  if (m_lerp == 0) { // Integer median size, use outside list for stats
    median = median_o;
    mean = (sum_o + sum_i) / max(med_n_o, 1);
    std_dev = sqrtf(max((sum_sq_o + sum_sq_i) / max(med_n_o, 1) - mean * mean, 0.0f));
  } else { // Lerp inner and outter stats
    // lerp between inner median and full median
    median = m_i_lerp * median_i + m_lerp * median_o;
    float mean_i = sum_i / max(med_n_i, 1);
    float std_dev_i = sqrtf(max(sum_sq_i / max(med_n_i, 1) - mean_i * mean_i, 0.0f));
    float mean_o = (sum_i + sum_o) / max(med_n_o, 1);
    mean = m_i_lerp * mean_i + m_lerp * mean_o;
    float std_dev_o = sqrtf(max((sum_sq_i + sum_sq_o) / max(med_n_o, 1) - mean_o * mean_o, 0.0f));
    std_dev = m_i_lerp * std_dev_i + m_lerp * std_dev_o;
  }
  // Multiply std_dev by sharpness_ parameter
  std_dev = afx::Clamp(sharpness * 3 * std_dev, 0.0f, 1.0f);
  // set output pointers
  float* row_out_ptr = (float*)((char*)row_ptr + x_cuda * sizeof(float));
  // Lerp median and input value by std_dev
  *row_out_ptr = (1.0f - std_dev) * median + std_dev * tex2D<float>(tex, x, y);
}

//median launcher
extern "C" void MedianCuda(cudaTextureObject_t tex, float* row_ptr, long int row_pitch, int med_size_i, int med_size_o, int med_n_i, int med_n_o, float m_lerp, float m_i_lerp, float sharpness, afx::Bounds img_bnds, afx::Bounds row_bnds, cudaStream_t stream) {

  int med_size_2 = med_size_o * 2;

  dim3 threads(256, 1);
  dim3 blocks((row_bnds.GetWidth() + threads.x - 1) / threads.x, (row_bnds.GetHeight() + threads.y - 1) / threads.y);

  KernelMedian<<<blocks, threads, 0, stream>>>(tex, row_ptr, row_pitch, med_size_i, med_size_o, med_n_i, med_n_o, m_lerp, m_i_lerp, sharpness, img_bnds, row_bnds, med_size_2);

}
