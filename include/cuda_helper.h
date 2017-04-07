// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_CUDA_HELPER_H_
#define AFX_CUDA_HELPER_H_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>

//*** for defintions
#include <cstdint>
#include <algorithm>
#include <list>
//***

#include "include/bounds.h"
#include "include/attribute.h"

#ifdef _WIN32
  #ifdef _MSC_VER
    #pragma warning(disable: 4251)
  #endif
  #define WINLIB_EXPORT __declspec(dllexport)
#else
  #define WINLIB_EXPORT
#endif

namespace afx {

// Cuda error handling functions
inline void CudaSafeCall(cudaError err) {
  if (err != cudaSuccess) { throw err; }
}
void CudaCheckError() {
  cudaError err = cudaGetLastError();
  CudaSafeCall(err);
}

class WINLIB_EXPORT CudaProcess {
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
class WINLIB_EXPORT CudaArray : public AttributeBase {
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
class WINLIB_EXPORT CudaImage : public AttributeBase {
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

class WINLIB_EXPORT CudaComplex : public AttributeBase {
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


class WINLIB_EXPORT CudaImageArray : public Array<CudaImage> {
 public:
  void AddImage(const Bounds& region);
  void CreateTextures();
};

class WINLIB_EXPORT CudaStream : public AttributeBase {
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


class WINLIB_EXPORT CudaStreamArray : public Array<CudaStream> {};





void CudaProcess::DevCount_() {
  CudaSafeCall(cudaGetDeviceCount(&dev_count_));
  if (dev_count_ < 1) { throw cudaErrorNoDevice; }
}
CudaProcess::CudaProcess() :ready_(false) {}
CudaProcess::~CudaProcess() {
  // TODO(rpw): Nuke hangs with these. Maybe not host thread safe. Look into it.
  // Synchonize();
  // Reset();
}
void CudaProcess::MakeReady() {
  DevCount_();
  SetFastestDevice();
  ready_ = true;
}
void CudaProcess::Synchonize() { CudaSafeCall(cudaDeviceSynchronize()); }
void CudaProcess::Reset() {
  ready_ = false;
  CudaSafeCall(cudaDeviceReset());
}
void CudaProcess::SetDevice(int id) {
  ready_ = false;
  if (id < dev_count_ - 1) { throw cudaErrorNoDevice; }
  dev_id_ = id;
  ready_ = true;
}
void CudaProcess::SetFastestDevice() {
  ready_ = false;
  // Compute capability required
  // Using Bindless textures that requires Kepler
  int major_req = 3;
  int minor_req = 0;

  float max_clock = 0;
  float clock = 0;
  int dev_num = -1;  // Initialize to -1 so we know if there are no suitatble devices

  // Find device with max flops
  cudaDeviceProp prop;
  for (int i = 0; i < dev_count_; ++i) {
    cudaGetDeviceProperties(&prop, i);
    clock = static_cast<float>(prop.multiProcessorCount) * prop.clockRate / 1000000.0f;
    if (prop.major >= major_req && prop.minor >= minor_req && clock > max_clock) {
      max_clock = clock;
      dev_num = i;
    }
  }
  if (dev_num == -1) { throw cudaErrorNotSupported; }
  dev_id_ = dev_num;
  prop_ = prop;
  ready_ = true;
}
int CudaProcess::GetDevice() const { return dev_id_; }
cudaDeviceProp CudaProcess::GetProp() const { return prop_; }
bool CudaProcess::CheckReady() const{
  if (!ready_) {
    throw cudaErrorDevicesUnavailable;
  }
  return true;
}
void CudaProcess::CheckError() { CudaCheckError(); }




template <typename T>
CudaArray<T>::CudaArray() : ptr_(nullptr) {}
template <typename T>
CudaArray<T>::CudaArray(size_t size) : ptr_(nullptr) { Create(size); }
template <typename T>
CudaArray<T>::CudaArray(const CudaArray<T>& other) : ptr_(nullptr) {
  Create(other.size);
  CudaSafeCall(cudaMemcpy(ptr_, other.GetPtr(), size_ * sizeof(T), cudaMemcpyHostToHost));
}
template <typename T>
CudaArray<T>& CudaArray<T>::operator=(const CudaArray<T>& other) {
  Create(other.size);
  CudaSafeCall(cudaMemcpy(ptr_, other.GetPtr(), size_ * sizeof(T), cudaMemcpyHostToHost));
  return *this;
}
template <typename T>
CudaArray<T>::~CudaArray() { Dispose(); }
template <typename T>
void CudaArray<T>::Create(size_t size) {
  Dispose();
  size_ = size;
  CudaSafeCall(cudaMalloc(reinterpret_cast<void**>(&ptr_), size_ * sizeof(T)));
}
template <typename T>
void CudaArray<T>::Dispose() {
  if (ptr_ != nullptr) {
    CudaSafeCall(cudaFree(ptr_) );
    ptr_ = nullptr;
  }
}
template <typename T>
void CudaArray<T>::MemSet(int value) {
  CudaSafeCall(cudaMemset(ptr_, value, size_ * sizeof(T)));
}
template <typename T>
void CudaArray<T>::MemSet(cudaStreamCallback_t stream, int value) {
  CudaSafeCall(cudaMemsetAsync(ptr_, value, size_ * sizeof(T), stream));
}
template <typename T>
void CudaArray<T>::MemCpyToDevice(T* h_ptr) {
  CudaSafeCall(cudaMemcpy(ptr_, h_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
void CudaArray<T>::MemCpyToDevice(T* h_ptr, cudaStream_t stream) {
  CudaSafeCall(cudaMemcpyAsync(ptr_, h_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
}
template <typename T>
void CudaArray<T>::MemCpyFromDevice(T* h_ptr) {
  CudaSafeCall(cudaMemcpy(h_ptr, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void CudaArray<T>::MemCpyFromDevice(T* h_ptr, cudaStream_t stream) {
  CudaSafeCall(cudaMemcpyAsync(h_ptr, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
}
template <typename T>
T* CudaArray<T>::GetPtr() const { return ptr_; }
template <typename T>
size_t CudaArray<T>::GetSize() const { return size_; }




CudaImage::CudaImage() : ptr_(nullptr), region_(Bounds()), has_tex_(false) {}
CudaImage::CudaImage(unsigned int width, unsigned int height) : ptr_(nullptr), has_tex_(false) { Create(Bounds(0, 0, width - 1, height - 1)); }
CudaImage::CudaImage(const Bounds& region) : ptr_(nullptr), has_tex_(false) { Create(region); }
CudaImage::CudaImage(const CudaImage& other) : ptr_(nullptr), has_tex_(false) {
  Create(other.region_);
  CudaSafeCall(cudaMemcpy2D(ptr_, pitch_, other.GetPtr(), other.GetPitch(), region_.GetWidth() * sizeof(float), region_.GetHeight(), cudaMemcpyHostToHost));
  if (other.has_tex_) { CreateTexture(other.address_mode_); }
}
CudaImage& CudaImage::operator=(const CudaImage& other) {
  Create(other.region_);
  CudaSafeCall(cudaMemcpy2D(ptr_, pitch_, other.GetPtr(), other.GetPitch(), region_.GetWidth() * sizeof(float), region_.GetHeight(), cudaMemcpyHostToHost));
  if (other.has_tex_) { CreateTexture(other.address_mode_); }
  return *this;
}
CudaImage::~CudaImage() { Dispose(); }
void CudaImage::Create(unsigned int width, unsigned int height) {
  Create(Bounds(0, width - 1, 0, height - 1));
}
void CudaImage::Create(const Bounds& region) {
  Dispose();  // Dispose before creating
  region_ = region;
  CudaSafeCall(cudaMallocPitch(reinterpret_cast<void**>(&ptr_), &pitch_, region_.GetWidth() * sizeof(float), region_.GetHeight()));
}
void CudaImage::Dispose() {
  if (ptr_ != nullptr) {
    CudaSafeCall(cudaFree(ptr_));
    ptr_ = nullptr;
  }
  DisposeTexture();
}
void CudaImage::CreateTexture(const cudaTextureAddressMode& address_mode) {
  DisposeTexture();  // Dispose of old texture
  address_mode_ = address_mode;  // Set private address mode -- how out of bound pixel addressing is handled
  memset(&res_desc_, 0, sizeof(res_desc_));  // memset res_desc_
  res_desc_.resType = cudaResourceTypePitch2D;  // set type to pitch2D for 2D image
  res_desc_.res.pitch2D.desc = cudaCreateChannelDesc<float>();  // Create channels
  res_desc_.res.pitch2D.devPtr = ptr_;  // Device image data pointer
  res_desc_.res.pitch2D.pitchInBytes = pitch_;  // device ptich
  res_desc_.res.pitch2D.width = region_.GetWidth();
  res_desc_.res.pitch2D.height = region_.GetHeight();
  memset(&tex_desc_, 0, sizeof(tex_desc_));  // mem set tex desc
  tex_desc_.readMode = cudaReadModeElementType;  // Set read mode to element, not normalized float.
  for (int i = 0; i < 3; ++i) { tex_desc_.addressMode[0] = address_mode; }  // Set address mode in all dimensions
  CudaSafeCall(cudaCreateTextureObject(&tex_, &res_desc_, &tex_desc_, NULL));  // Create texture object
  has_tex_ = true;
}
void CudaImage::DisposeTexture() {
  if (has_tex_) {
    CudaSafeCall(cudaDestroyTextureObject(tex_));
    has_tex_ = false;
  }
}
void CudaImage::MemSet(int value) {
  CudaSafeCall(cudaMemset2D(ptr_, pitch_, value, region_.GetWidth() * sizeof(float), region_.GetHeight()));
}
void CudaImage::MemSet(cudaStream_t stream, int value) {
  CudaSafeCall(cudaMemset2DAsync(ptr_, pitch_, value, region_.GetWidth() * sizeof(float), region_.GetHeight(), stream));
}
void CudaImage::MemCpyToDevice(const float* h_ptr, size_t h_pitch) {
  CudaSafeCall(cudaMemcpy2D(ptr_, pitch_, h_ptr, h_pitch, region_.GetWidth() * sizeof(float), region_.GetHeight(), cudaMemcpyHostToDevice));
}
void CudaImage::MemCpyToDevice(const float* h_ptr, size_t h_pitch, cudaStream_t stream) {
  CudaSafeCall(cudaMemcpy2DAsync(ptr_, pitch_, h_ptr, h_pitch, region_.GetWidth() * sizeof(float), region_.GetHeight(), cudaMemcpyHostToDevice, stream));
}
void CudaImage::MemCpyFromDevice(float* h_ptr, size_t h_pitch) {
  CudaSafeCall(cudaMemcpy2D(h_ptr, h_pitch, ptr_, pitch_, region_.GetWidth() * sizeof(float), region_.GetHeight(), cudaMemcpyDeviceToHost));
}
void CudaImage::MemCpyFromDevice(float* h_ptr, size_t h_pitch, cudaStream_t stream) {
  CudaSafeCall(cudaMemcpy2DAsync(h_ptr, h_pitch, ptr_, pitch_, region_.GetWidth() * sizeof(float), region_.GetHeight(), cudaMemcpyDeviceToHost, stream));
}
float* CudaImage::GetPtr() const { return ptr_; }
float* CudaImage::GetPtr(int x, int y) const {
  float* ptr = reinterpret_cast<float*>(reinterpret_cast<std::uint8_t*>(ptr_) + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(float));
  return ptr;
}
size_t CudaImage::GetPitch() const { return pitch_; }
unsigned int CudaImage::GetWidth() const { return region_.GetWidth(); }
unsigned int CudaImage::GetHeight() const { return region_.GetHeight(); }
Bounds CudaImage::GetBounds() const { return region_; }
bool CudaImage::HasTexture() const { return has_tex_; }
cudaTextureObject_t CudaImage::GetTexture() const {
  if (has_tex_) {
    return tex_;
  } else {
    throw cudaErrorTextureFetchFailed;
  }
}




CudaComplex::CudaComplex() : ptr_(nullptr) {}
CudaComplex::CudaComplex(unsigned int width, unsigned int height) : ptr_(nullptr) { Create(width, height); }
CudaComplex::CudaComplex(const CudaComplex& other) : ptr_(nullptr) {
  Create(other.GetHeight(), other.GetHeight());
  CudaSafeCall(cudaMemcpy2D(ptr_, pitch_, other.GetPtr(), other.GetPitch(), width_ * sizeof(cufftComplex), height_, cudaMemcpyHostToHost));
}
CudaComplex& CudaComplex::operator= (const CudaComplex& other) {
  Create(other.GetHeight(), other.GetHeight());
  CudaSafeCall(cudaMemcpy2D(ptr_, pitch_, other.GetPtr(), other.GetPitch(), width_ * sizeof(cufftComplex), height_, cudaMemcpyHostToHost));
  return *this;
}
CudaComplex::~CudaComplex() { Dispose(); }
void CudaComplex::Create(unsigned int width, unsigned int height) {
  Dispose();  // Dispose before creating
  width_ = width;
  height_ = height;
  CudaSafeCall(cudaMallocPitch(reinterpret_cast<void**>(&ptr_), &pitch_, width_ * sizeof(cufftComplex), height_));
}
void CudaComplex::Dispose() {
  if (ptr_ != nullptr) {
    CudaSafeCall(cudaFree(ptr_));
    ptr_ = nullptr;
  }
}
cufftComplex* CudaComplex::GetPtr() const { return ptr_; }
cufftComplex* CudaComplex::GetPtr(int x, int y) const {
  cufftComplex* ptr = reinterpret_cast<cufftComplex*>(reinterpret_cast<std::uint8_t*>(ptr_) + y * pitch_ + x * sizeof(cufftComplex));
  return ptr;
}
size_t CudaComplex::GetPitch() const {return pitch_; }
unsigned int CudaComplex::GetWidth() const { return width_; }
unsigned int CudaComplex::GetHeight() const { return height_; }






void CudaImageArray::AddImage(const Bounds& region) { this->array_.push_back(new CudaImage(region)); }
void CudaImageArray::CreateTextures() {
  for (ptr_list_it it = array_.begin(); it != array_.end(); ++it) {
    it->CreateTexture();
  }
}



CudaStream::CudaStream() { CudaSafeCall(cudaStreamCreate(&stream_)); }
CudaStream::CudaStream(const CudaStream& other) {
  stream_ = other.stream_;
  attributes_ = other.attributes_;
}
CudaStream& CudaStream::operator=(const CudaStream& other) {
  stream_ = other.stream_;
  attributes_ = other.attributes_;
  return *this;
}
CudaStream::~CudaStream() { CudaSafeCall(cudaStreamDestroy(stream_)); }
void CudaStream::Synchonize() { CudaSafeCall(cudaStreamSynchronize(stream_)); }
cudaStream_t CudaStream::GetStream() { return stream_; }


}  // namespace afx

#endif  // AFX_CUDA_HELPER_H_
