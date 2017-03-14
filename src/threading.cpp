// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#include "include/threading.h"

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/locks.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>

#include <algorithm>

namespace afx {

Threader::Threader() : running_(false) { InitializeThreads(0); }
Threader::Threader(unsigned int req_num_threads) : running_(false) { InitializeThreads(req_num_threads); }
Threader::~Threader() { StopAndJoin(); }
void Threader::AddThreads(unsigned int num_threads) {
  for (unsigned int t = 0; t < num_threads; ++t) {
    thread_pool_.create_thread(boost::bind(&Threader::Worker_, this));
  }
}
void Threader::InitializeThreads(unsigned int req_num_threads) {
  if (running_) { StopAndJoin(); }  // If running join threads so empty thread pool
  work_ptr_.reset(new boost::asio::io_service::work(io_service_));
  if (io_service_.stopped()) { io_service_.reset(); }
  running_ = true;
  unsigned int avail_threads = boost::thread::hardware_concurrency();
  num_threads_ = req_num_threads > 0 ? std::min<unsigned int>(req_num_threads, avail_threads) : avail_threads;
  for (unsigned int t = 0; t < num_threads_; ++t) {
    thread_pool_.create_thread(boost::bind(&Threader::Worker_, this));
  }
}
void Threader::Worker_() {
  while (running_) {
    io_service_.run();  // Blocks until work is complete
    exited_run_.notify_one();  // Notify that this thread has exited run()
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (running_ && io_service_.stopped()) {
      io_service_ready_.wait(lock);  // Wait until io_service_ has been reset
    }
  }
}
void Threader::Wait() {
  work_ptr_.reset();  // Destroy work object to allow run all handlers to finish normally and for run to return.
  boost::mutex dummy_mutex;
  boost::unique_lock<boost::mutex> dummy_lock(dummy_mutex);
  while (!io_service_.stopped()) {
    exited_run_.wait(dummy_lock);  // Wait for all threads to exit run() in Worker_.
  }
  if (running_) {
    boost::lock_guard<boost::mutex> lock(mutex_);
    work_ptr_.reset(new boost::asio::io_service::work(io_service_));
    io_service_.reset();
  }
  io_service_ready_.notify_all();  // Allow threads to advance to end of while loop in Worker_
}
void Threader::StopAndJoin() {
  {
    boost::lock_guard<boost::mutex> lock(mutex_);
    running_ = false;
  }
  Wait();
  thread_pool_.join_all();
}
void Threader::AddWork(boost::function<void()> function) { io_service_.post(function); }
bool Threader::IsRunning() const { return running_; }
unsigned int Threader::Threads() const { return num_threads_; }

void ImageThreader::ThreadImageRows(const Bounds& region, boost::function<void(Bounds)> function) {
  Bounds thread_region = region;
  for (int row = region.y1(); row <= region.y2(); ++row) {
    thread_region.SetY(row, row);
    AddWork(boost::bind(function, thread_region));
  }
}
void ImageThreader::ThreadImageColumns(const Bounds& region, boost::function<void(Bounds)> function) {
  Bounds thread_region = region;
  for (int column = region.x1(); column <= region.x2(); ++column) {
    thread_region.SetX(column, column);
    AddWork(boost::bind(function, thread_region));
  }
}
void ImageThreader::ThreadImageChunks(const Bounds& region, boost::function<void(Bounds)> function) {
  unsigned int num_chunks = Threads();
  num_chunks = std::min(num_chunks, region.GetHeight());
  Bounds thread_region = region;
  for (int i = 0; i < num_chunks; ++i) {
    thread_region.SetY1(static_cast<int>(ceil(static_cast<float>(region.GetHeight()) * static_cast<float>(i)     / static_cast<float>(num_chunks))      + region.y1()));
    thread_region.SetY2(static_cast<int>(ceil(static_cast<float>(region.GetHeight()) * static_cast<float>(i + 1) / static_cast<float>(num_chunks)) - 1  + region.y1()));
    AddWork(boost::bind(function, thread_region));
  }
}
void ImageThreader::ThreadImageChunksY(const Bounds& region, boost::function<void(Bounds)> function) {
  unsigned int num_chunks = Threads();
  num_chunks = std::min(num_chunks, region.GetWidth());
  Bounds thread_region = region;
  for (int i = 0; i < num_chunks; ++i) {
    thread_region.SetX1(static_cast<int>(ceil(static_cast<float>(region.GetWidth()) * static_cast<float>(i)     / static_cast<float>(num_chunks))      + region.x1()));
    thread_region.SetX2(static_cast<int>(ceil(static_cast<float>(region.GetWidth()) * static_cast<float>(i + 1) / static_cast<float>(num_chunks)) - 1  + region.x1()));
    AddWork(boost::bind(function, thread_region));
  }
}

}  // namespace afx
