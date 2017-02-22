// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_MEDIAN_H_
#define INCLUDE_MEDIAN_H_

namespace afx {

__host__ __device__
void Swap(float* a, float* b) {
  float* temp = a;
  a = b;
  b = temp;
}

__host__ __device__
void Sort(float* a, float* b) {
  if (*a > *b) { Swap(a, b); }
}

__host__ __device__
float MedQuick9(float* p) {
  Sort(&p[1], &p[2]); Sort(&p[4], &p[5]); Sort(&p[7], &p[8]);
  Sort(&p[0], &p[1]); Sort(&p[3], &p[4]); Sort(&p[6], &p[7]);
  Sort(&p[1], &p[2]); Sort(&p[4], &p[5]); Sort(&p[7], &p[8]);
  Sort(&p[0], &p[3]); Sort(&p[5], &p[8]); Sort(&p[4], &p[7]);
  Sort(&p[3], &p[6]); Sort(&p[1], &p[4]); Sort(&p[2], &p[5]);
  Sort(&p[4], &p[7]); Sort(&p[4], &p[2]); Sort(&p[6], &p[4]);
  Sort(&p[4], &p[2]);
  return(p[4]);
}

__host__ __device__
float MedQuick25(float* p) {
  Sort(&p[0],  &p[1]);  Sort(&p[3],  &p[4]);  Sort(&p[2],  &p[4]);
  Sort(&p[2],  &p[3]);  Sort(&p[6],  &p[7]);  Sort(&p[5],  &p[7]);
  Sort(&p[5],  &p[6]);  Sort(&p[9],  &p[10]); Sort(&p[8],  &p[10]);
  Sort(&p[8],  &p[9]);  Sort(&p[12], &p[13]); Sort(&p[11], &p[13]);
  Sort(&p[11], &p[12]); Sort(&p[15], &p[16]); Sort(&p[14], &p[16]);
  Sort(&p[14], &p[15]); Sort(&p[18], &p[19]); Sort(&p[17], &p[19]);
  Sort(&p[17], &p[18]); Sort(&p[21], &p[22]); Sort(&p[20], &p[22]);
  Sort(&p[20], &p[21]); Sort(&p[23], &p[24]); Sort(&p[2],  &p[5]);
  Sort(&p[3],  &p[6]);  Sort(&p[0],  &p[6]);  Sort(&p[0],  &p[3]);
  Sort(&p[4],  &p[7]);  Sort(&p[1],  &p[7]);  Sort(&p[1],  &p[4]);
  Sort(&p[11], &p[14]); Sort(&p[8],  &p[14]); Sort(&p[8],  &p[11]);
  Sort(&p[12], &p[15]); Sort(&p[9],  &p[15]); Sort(&p[9],  &p[12]);
  Sort(&p[13], &p[16]); Sort(&p[10], &p[16]); Sort(&p[10], &p[13]);
  Sort(&p[20], &p[23]); Sort(&p[17], &p[23]); Sort(&p[17], &p[20]);
  Sort(&p[21], &p[24]); Sort(&p[18], &p[24]); Sort(&p[18], &p[21]);
  Sort(&p[19], &p[22]); Sort(&p[8],  &p[17]); Sort(&p[9],  &p[18]);
  Sort(&p[0],  &p[18]); Sort(&p[0],  &p[9]);  Sort(&p[10], &p[19]);
  Sort(&p[1],  &p[19]); Sort(&p[1],  &p[10]); Sort(&p[11], &p[20]);
  Sort(&p[2],  &p[20]); Sort(&p[2],  &p[11]); Sort(&p[12], &p[21]);
  Sort(&p[3],  &p[21]); Sort(&p[3],  &p[12]); Sort(&p[13], &p[22]);
  Sort(&p[4],  &p[22]); Sort(&p[4],  &p[13]); Sort(&p[14], &p[23]);
  Sort(&p[5],  &p[23]); Sort(&p[5],  &p[14]); Sort(&p[15], &p[24]);
  Sort(&p[6],  &p[24]); Sort(&p[6],  &p[15]); Sort(&p[7],  &p[16]);
  Sort(&p[7],  &p[19]); Sort(&p[13], &p[21]); Sort(&p[15], &p[23]);
  Sort(&p[7],  &p[13]); Sort(&p[7],  &p[15]); Sort(&p[1],  &p[9]);
  Sort(&p[3],  &p[11]); Sort(&p[5],  &p[17]); Sort(&p[11], &p[17]);
  Sort(&p[9],  &p[17]); Sort(&p[4],  &p[10]); Sort(&p[6],  &p[12]);
  Sort(&p[7],  &p[14]); Sort(&p[4],  &p[6]);  Sort(&p[4],  &p[7]);
  Sort(&p[12], &p[14]); Sort(&p[10], &p[14]); Sort(&p[6],  &p[7]);
  Sort(&p[10], &p[12]); Sort(&p[6],  &p[10]); Sort(&p[6],  &p[17]);
  Sort(&p[12], &p[17]); Sort(&p[7],  &p[17]); Sort(&p[7],  &p[10]);
  Sort(&p[12], &p[18]); Sort(&p[7],  &p[12]); Sort(&p[10], &p[18]);
  Sort(&p[12], &p[20]); Sort(&p[10], &p[20]); Sort(&p[10], &p[12]);
  return (p[12]);
}

// This Quickselect routine is based on the algorithm described in
// "Numerical recipes in C", Second Edition,
// Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
// This code by Nicolas Devillard - 1998. Public domain.
__host__ __device__
float MedianQuickSelect(float* list, int n) {
  switch (n) {
    case 9: { return MedQuick9(list); }
    case 25: { return MedQuick25(list); }
  }

  int low, high;
  int median;
  int middle, ll, hh;

  low = 0;
  high = n - 1;
  median = (low + high) / 2;
  while (1) {
    if (high <= low) { return list[median]; }  // One element only
    if (high == low + 1) {  // Two elements only
      if (list[low] > list[high]) {
        Swap(&list[low], &list[high]);
        return list[median];
      }
    }

    // Find median of low, middle and high items; swap into position low
    middle = (low + high) / 2;
    if (list[middle] > list[high]) { Swap(&list[middle], &list[high]); }
    if (list[low] > list[high])    { Swap(&list[low], &list[high]); }
    if (list[middle] > list[low])  { Swap(&list[middle], &list[low]); }

    // Swap low item (now in position middle) into position (low + 1)
    Swap(&list[middle], &list[low + 1]);

    // Nibble from each end towards middle, swapping items when stuck
    ll = low + 1;
    hh = high;
    while (1) {
      do { ll++; } while (list[low] > list[ll]);
      do { hh--; } while (list[hh]  > list[low]);
      if (hh < ll) { break; }
      Swap(&list[ll], &list[hh]);
    }

    // Swap middle item (in position low) back into correct position
    Swap(&list[low], &list[hh]);

    // Re-set active partition
    if (hh <= median) { low = ll; }
    if (hh >= median) { high = hh - 1; }
  }
}

}  // namespace afx

#endif  // INCLUDE_MEDIAN_H_
