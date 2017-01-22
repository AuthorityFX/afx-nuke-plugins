// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef COLOR_OP_H_
#define COLOR_OP_H_

#include <algorithm>
#include <math.h>

namespace afx {

enum SuppresionAlgorithm {
  kLight = 0,
  kMedium = 1,
  kHard = 2,
  kHarder = 3,
  kEvenHarder = 4,
  kExtreme = 5,
  kUber = 6
};

template <typename T>
inline T AfxClamp(T value, T min, T max) {
  return std::max(std::min(value, max), min);
}
inline float min3(const float& a, const float& b, const float& c) {
  return fminf(a, fminf(b, c));
}
inline float max3(const float& a, const float& b, const float& c) {
  return fmaxf(a, fmaxf(b, c));
}
inline void RGBtoHSV(const float (&RGB)[3], float* HSV);
inline void HSVtoRGB(const float (&HSV)[3], float* RGB);
inline float SpillSuppression(const float (&RGB)[3], int algorithm);

inline float SoftClip(float value, float clip, float knee) {
  if (value <= 0.0f) { return value; }
  knee = AfxClamp(knee, 0.008f, 125.0f);
  return clip * powf(tanhf(powf(value / clip, knee)), 1.0f / knee);
}

class RotateColor {
 private:
  float m_[3][3];
 public:
  RotateColor();
  // Create transform object. Use Apply to perform an inplace rotation.
  RotateColor(float deg);
  // Create transform object. Use Apply to perform an inplace rotation.
  RotateColor(const float (&axis)[3], float deg);
  // Build transform matrix for rotation
  inline void BuildMatrix(const float (&axis)[3], float deg);
  // Build transform matrix for rotation, TODO I don't remember why this
  // only has rotation and no axis.
  inline void BuildMatrix(float deg);
  // Apply in place color rotation
  inline void Rotate(float* color);
  inline void Invert();
  // Get transform matrix
  inline float GetMatrix(int r, int c);
};

RotateColor::RotateColor() {
  //Initialize identity matrix.
  for (int r=0; r<3; r++) {
    for (int c=0; c<3; c++) {
      if (c==r) {
        m_[r][c] = 1.0f;
      }
      else {
        m_[r][c] = 0.0f;
      }
    }
  }
}
RotateColor::RotateColor(float deg) {
  BuildMatrix(deg);
}
RotateColor::RotateColor(const float (&axis)[3], float deg) {
  BuildMatrix(axis, deg);
}
void RotateColor::BuildMatrix(const float (&axis)[3], float deg) {
  float angle = deg * M_PI / 180.0f;
  float cosA = cos(angle);
  float sinA = sin(angle);
  float x = axis[0];
  float y = axis[1];
  float z = axis[2];
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float xy = x * y;
  float xz = x * z;
  float yz = y * z;
  //Rodrigues' rotation formula
  //http://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
  m_[0][0] = cosA + (xx) * (1 - cosA);
  m_[0][1] = (xy) * (1 - cosA) - (z) * sinA;
  m_[0][2] = (xz) * (1 - cosA) + (y) * sinA;
  m_[1][0] = (xy) * (1 - cosA) + (z) * sinA;
  m_[1][1] = cosA + (yy) * (1 - cosA);
  m_[1][2] = -(x) * sinA + (yz) * (1 - cosA);
  m_[2][0] = (xz) * (1 - cosA) - (y) * sinA;
  m_[2][1] = (yz) * (1 - cosA) + (x) * sinA;
  m_[2][2] = cosA + (zz) * (1 - cosA);
}
void RotateColor::BuildMatrix(float deg) {
  float angle = deg * M_PI / 180.0f;
  float cosA = cos(angle);
  float sinA = sin(angle);
  float x = 1.0f / sqrt(3.0f);
  float xx = 1.0f / 3.0f;
  m_[0][0] = cosA + (xx) * (1 - cosA);
  m_[0][1] = (xx) * (1 - cosA) - (x) * sinA;
  m_[0][2] = (xx) * (1 - cosA) + (x) * sinA;
  m_[1][0] = (xx) * (1 - cosA) + (x) * sinA;
  m_[1][1] = cosA + (xx)*(1 - cosA);
  m_[1][2] = -(x) * sinA + (xx) * (1 - cosA);
  m_[2][0] = (xx) * (1 - cosA) - (x) * sinA;
  m_[2][1] = (xx) * (1 - cosA) + (x) * sinA;
  m_[2][2] = cosA + (xx) * (1 - cosA);
}
void RotateColor::Rotate(float* color) {
  float temp[3] = {0.0f, 0.0f, 0.0f};
  for (int r=0; r<3; r++) {
    for (int c=0; c<3; c++) {
      temp[r] += m_[r][c] * color[c];
    }
  }
  for (int i=0; i<3; i++) {
    color[i] = temp[i];
  }
}
void RotateColor::Invert() {
  for (int r=0; r<3; r++) {
    for (int c=r; c<3; c++) {
      float temp;
      temp = m_[r][c];
      m_[r][c] = m_[c][r];
      m_[c][r] = temp;
    }
  }
}
float RotateColor::GetMatrix(int r, int c) {
  return m_[r][c];
}


void RGBtoHSV(const float (&rgb)[3], float* hsv) {
    float a = 0.5f * (2 * rgb[0] - rgb[1] - rgb[2]);
    float b = (sqrtf(3) / 2) * (rgb[1] - rgb[2]);
    hsv[0] = atan2f(b, a) / (2.0f * M_PI);
    hsv[0] = fmod(hsv[0], 1.0f);
    if (hsv[0] < 0) { hsv[0] += 1.0f; }
    hsv[1] = sqrtf(powf(a, 2.0f) + powf(b, 2.0f));
    hsv[2] = max3(rgb[0], rgb[1], rgb[2]);
}
void HSVtoRGB(const float (&HSV)[3], float* RGB)
{
  float C = HSV[2] * HSV[1];
  double Ho = 6.0f * HSV[0];
  float X = C * (1.0f - fabsf((float)fmodf(Ho, 2.0) - 1.0f));
  if( Ho >= 0 && Ho < 1) {
    RGB[0] = C;
    RGB[1] = X;
    RGB[2] = 0;
  } else if( Ho >= 1 && Ho < 2) {
    RGB[0] = X;
    RGB[1] = C;
    RGB[2] = 0;
  } else if( Ho >= 2 && Ho < 3) {
    RGB[0] = 0;
    RGB[1] = C;
    RGB[2] = X;
  } else if( Ho >= 3 && Ho < 4) {
    RGB[0] = 0;
    RGB[1] = X;
    RGB[2] = C;
  } else if( Ho >= 4 && Ho < 5) {
    RGB[0] = X;
    RGB[1] = 0;
    RGB[2] = C;
  } else if( Ho >= 5) {
    RGB[0] = C;
    RGB[1] = 0;
    RGB[2] = X;
  }
  float m = HSV[2] - C;
  RGB[0] += m;
  RGB[1] += m;
  RGB[2] += m;
}

float f(float t) {
  if (t > powf(6.0f / 29.0f, 3.0f)) {
    return powf(t, 1.0f / 3.0f);
  } else {
    return (1.0f / 3.0f) * powf(29.0f / 6.0f, 2.0f) * t + (4.0f / 29.0f);
  }
}
float f_inv(float t) {
  if (t > 1000000) {
    return powf(t, 3.0f);
  } else {
    return 3 * powf(6.0f / 29.0f, 2.0f) * (t - (4.0f / 29.0f));
  }
}
void RGBtoLab(const float (&rgb)[3], float (&lab)[3]) {
  float X = 41.2453f * rgb[0] + 35.7580f * rgb[1] + 18.0423f * rgb[2];
  float Y = 21.2671f * rgb[0] + 71.5160f * rgb[1] + 7.2169f  * rgb[2];
  float Z = 1.9334f  * rgb[0] + 11.9193f * rgb[1] + 95.0227f * rgb[2];
  lab[0] = 116 *  f(Y / 100.0f)  - 16;
  lab[1] = 500 * (f(X / 95.047f) - f(Y / 100.0f));
  lab[2] = 200 * (f(Y / 100.0f)  - f(Z / 108.883f));
}
void LabtoRGB(const float (&lab)[3], float (&rgb)[3]) {
  //https://en.wikipedia.org/wiki/Lab_color_space
  float X = 95.047f  * f_inv((1.0f / 116.0f)*(lab[0] + 16.0f) + (1.0f / 500.0f) * lab[1]);
  float Y = 100.0f   * f_inv((1.0f / 116.0f)*(lab[0] + 16.0f));
  float Z = 108.883f * f_inv((1.0f / 116.0f)*(lab[0] + 16.0f) + (1.0f / 200.0f) * lab[2]);
}

float SpillSuppression(const float (&RGB)[3], int algorithm) {
  float temp = 0.0f;
  float suppression = 0.0f;

  switch (algorithm) {
    case kLight: {
      temp = RGB[1] - fmaxf(RGB[0], RGB[2]);
      if (temp > 0) { suppression = temp; }
      break;
    }
    case kMedium: {
      temp = 2 * RGB[1] - fmaxf(RGB[0], RGB[2]) - (RGB[0] + RGB[2]) / 2.0f;
      if (temp > 0) { suppression = temp / 2.0f; }
      break;
    }
    case kHard: {
      temp = RGB[1] - (RGB[0] + RGB[2]) / 2.0f;
      if (temp > 0) { suppression = temp; }
      break;
    }
    case kHarder: {
      temp = 3 * RGB[1] - (RGB[0] + RGB[2]) - fminf(RGB[0], RGB[2]);
      if (temp > 0) { suppression = temp / 3.0f; }
      break;
    }
    case kEvenHarder: {
      temp = 2 * RGB[1] - (RGB[0] + RGB[2]) / 2.0f - fminf(RGB[0], RGB[2]);
      if (temp > 0) { suppression = temp / 2.0f; }
      break;
    }
    case kExtreme: {
      temp = 3 * RGB[1] - (RGB[0] + RGB[2]) / 2.0f - 2 * fminf(RGB[0], RGB[2]);
      if (temp > 0) { suppression = temp / 3.0f; }
      break;
    }
    case kUber: {
      temp = 4 * RGB[1] - (RGB[0] + RGB[2]) / 2.0f - 2 * fminf(RGB[0], RGB[2]);
      if (temp > 0) { suppression = temp / 4.0f; }
      break;
    }
  }
  return suppression;
}

} // namespace afx

#endif // COLOR_OP_H_
