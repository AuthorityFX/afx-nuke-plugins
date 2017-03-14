// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_CUDA_HELPER_H_
#define INCLUDE_CUDA_HELPER_H_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "include/bounds.h"
#include "include/attribute.h"

namespace afx {

// Cuda error handling functions
inline void CudaSafeCall(cudaError err) {
  if (err != cudaSuccess) { throw err; }
}
void CudaCheckError() {
  cudaError err = cudaGetLastError();
  CudaSafeCall(err);
}

class CudaProcess {
 private:
  int dev_id_;
  int dev_count_;
  cudaDeviceProp prop_;
  bool ready_;
  void DevCount_();

 public:
  CudaProcess();
  ~CudaProcess();
  void MakeReady();
  void Synchonize();
  void Reset();
  void SetDevice(int id);
  void SetFastestDevice();
  int GetDevice() const;
  cudaDeviceProp GetProp() const;
  bool CheckReady() const;
  void CheckError();
};

// Cuda device array wrapper class
template <typename T>
class CudaArray : public AttributeBase {
 private:
  T*  ptr_;
  size_t size_;
 public:
  CudaArray();
  explicit CudaArray(size_t size);
  CudaArray(const CudaArray<T>& other);
  CudaArray<T>& operator=(const CudaArray<T>& other);
  ~CudaArray();
  void Create(size_t size);
  void Dispose();
  void MemSet(int value = 0);
  void MemSet(cudaStreamCallback_t stream, int value = 0);
  void MemCpyToDevice(T* h_ptr);
  void MemCpyToDevice(T* h_ptr, cudaStream_t stream);
  void MemCpyFromDevice(T* h_ptr);
  void MemCpyFromDevice(T* h_ptr, cudaStream_t stream);
  T* GetPtr() const;
  size_t GetSize() const;
};

// Cuda image wrapper class
class CudaImage : public AttributeBase {
 private:
  // Image members
  float* ptr_;
  size_t pitch_;
  Bounds region_;
  // Texture members
  bool has_tex_;
  cudaResourceDesc res_desc_;
  cudaTextureDesc tex_desc_;
  cudaTextureObject_t tex_;
  cudaTextureAddressMode address_mode_;

 public:
  CudaImage();
  CudaImage(unsigned int width, unsigned int height);
  explicit CudaImage(const Bounds& region);
  CudaImage(const CudaImage& other);
  CudaImage& operator=(const CudaImage& other);
  ~CudaImage();
  void Create(unsigned int width, unsigned int height);
  void Create(const Bounds& region);
  void Dispose();
  void CreateTexture(const cudaTextureAddressMode& address_mode = cudaAddressModeClamp);
  void DisposeTexture();
  void MemSet(int value = 0);
  void MemSet(cudaStream_t stream, int value = 0);
  void MemCpyToDevice(const float* h_ptr, size_t h_pitch);
  void MemCpyToDevice(const float* h_ptr, size_t h_pitch, cudaStream_t stream);
  void MemCpyFromDevice(float* h_ptr, size_t h_pitch);
  void MemCpyFromDevice(float* h_ptr, size_t h_pitch, cudaStream_t stream);
  float* GetPtr() const;
  float* GetPtr(int x, int y) const;
  size_t GetPitch() const;
  unsigned int GetWidth() const;
  unsigned int GetHeight() const;
  Bounds GetBounds() const;
  bool HasTexture() const;
  cudaTextureObject_t GetTexture() const;
};

class CudaComplex : public AttributeBase {
 private:
  cufftComplex* ptr_;
  size_t pitch_;
  unsigned int width_;
  unsigned int height_;

 public:
  CudaComplex();
  CudaComplex(unsigned int width, unsigned int height);
  CudaComplex(const CudaComplex& other);
  CudaComplex& operator=(const CudaComplex& other);
  ~CudaComplex();
  void Create(unsigned int width, unsigned int height);
  void Dispose();
  cufftComplex* GetPtr() const;
  cufftComplex* GetPtr(int x, int y) const;
  size_t GetPitch() const;
  unsigned int GetWidth() const;
  unsigned int GetHeight() const;
};


class CudaImageArray : public Array<CudaImage> {
 public:
  void AddImage(const Bounds& region);
  void CreateTextures();
};

class CudaStream : public AttributeBase {
 private:
  cudaStream_t stream_;
 public:
  CudaStream();
  CudaStream(const CudaStream& other);
  CudaStream& operator=(const CudaStream& other);
  ~CudaStream();
  void Synchonize();
  cudaStream_t GetStream();
};


class CudaStreamArray : public Array<CudaStream> {};

}  // namespace afx

#endif  // INCLUDE_CUDA_HELPER_H_
