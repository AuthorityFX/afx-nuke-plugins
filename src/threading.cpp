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
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>

#include <algorithm>

#include "include/settings.h"

namespace afx {

Threader::Threader() : running_(false) { InitializeThreads(0); }
Threader::Threader(unsigned int req_num_threads) : running_(false) { InitializeThreads(req_num_threads); }
Threader::~Threader() { StopAndJoin(); }
bool Threader::InteruptionPoint() {
  try {
    boost::this_thread::interruption_point();
    return true;
  } catch (boost::thread_interrupted&) {
    return false;
  }
}
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
    io_service_.run();  // Blocks. Will do work from queue
    synchronized_.notify_one();  // Notify that this thread has exited run()
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (running_ && io_service_.stopped()) {
      ready_condition_.wait(lock);  // Wait until io_service_ has been reset
    }
  }
}
void Threader::Synchonize() {
  work_ptr_.reset();  // Destroy work object to allow run all handlers to finish normally and for run to return.
  boost::mutex dummy;
  boost::unique_lock<boost::mutex> lock(dummy);
  while (!io_service_.stopped()) {
    synchronized_.wait(lock);  // Wait for all threads to exit run() in Worker_.
  }
  if (running_) {
    work_ptr_.reset(new boost::asio::io_service::work(io_service_));
    io_service_.reset();
  }
  ready_condition_.notify_all();  // Allow threads to advance to end of while loop in Worker_
}
void Threader::Interupt() {  // TODO(rpw): This needs testing.
  running_ = false;
  thread_pool_.interrupt_all();
  io_service_.stop();  // Return from run asap
  Synchonize();
  thread_pool_.join_all();
}
void Threader::StopAndJoin() {
  running_ = false;
  Synchonize();
  thread_pool_.join_all();
}
void Threader::AddWork(boost::function<void()> function) { io_service_.post(function); }
void Threader::ThreadImageRows(const Bounds& region, boost::function<void(Bounds)> function) {
  Bounds thread_region = region;
  for (int row = region.y1(); row <= region.y2(); ++row) {
    thread_region.SetY(row, row);
    io_service_.post(boost::bind(function, thread_region));
  }
}
void Threader::ThreadImageChunks(const Bounds& region, boost::function<void(Bounds)> function) {
  unsigned int num_chunks = num_threads_;
  num_chunks = std::min(num_chunks, region.GetHeight());
  Bounds thread_region = region;
  for (int i = 0; i < num_chunks; ++i) {
    thread_region.SetY1(static_cast<int>(ceil(static_cast<float>(region.GetHeight()) * static_cast<float>(i)     / static_cast<float>(num_chunks))      + region.y1()));
    thread_region.SetY2(static_cast<int>(ceil(static_cast<float>(region.GetHeight()) * static_cast<float>(i + 1) / static_cast<float>(num_chunks)) - 1  + region.y1()));
    io_service_.post(boost::bind(function, thread_region));
  }
}
void Threader::ThreadImageChunksY(const Bounds& region, boost::function<void(Bounds)> function) {
  unsigned int num_chunks = num_threads_;
  num_chunks = std::min(num_chunks, region.GetWidth());
  Bounds thread_region = region;
  for (int i = 0; i < num_chunks; ++i) {
    thread_region.SetX1(static_cast<int>(ceil(static_cast<float>(region.GetWidth()) * static_cast<float>(i)     / static_cast<float>(num_chunks))      + region.x1()));
    thread_region.SetX2(static_cast<int>(ceil(static_cast<float>(region.GetWidth()) * static_cast<float>(i + 1) / static_cast<float>(num_chunks)) - 1  + region.x1()));
    io_service_.post(boost::bind(function, thread_region));
  }
}

}  // namespace afx
