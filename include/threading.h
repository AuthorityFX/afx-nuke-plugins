// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef THREADING_H_
#define THREADING_H_

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>

#include "types.h"

#include "settings.h"

namespace afx {

class Threader {
private:
  boost::asio::io_service io_service_;
  boost::thread_group thread_pool_;
  boost::scoped_ptr<boost::asio::io_service::work> work_ptr_;
  unsigned int num_threads_;

  boost::condition_variable synchronized_;
  boost::condition_variable ready_condition_;
  bool running_;

  boost::mutex mutex_;

  // This is the function that is run in each thread
  void Worker_();
public:
  Threader();
  Threader(unsigned int req_num_threads);
  ~Threader();
  // This is an exit point for Interupt()
  // Call this from function sent to Threader via AddWork.
  bool InteruptionPoint();
  // Launch additional threads.
  void AddThreads(unsigned int num_threads);
  // Start asio service. Launch req num of threads. 0 will launch hardware concurency
  void InitializeThreads(unsigned int req_num_threads = 0);
  // Poll until all work is submitted. Stop asio service. Block until threads have completed work. Restart asio service
  void Synchonize();
  // Exit threads at next boost::this_thread::interruption_point()
  void Interupt();
  // Synchonize and join all threads.
  void StopAndJoin();
  void AddWork(boost::function<void()> function);
  // Split bounds into num of rows.
  void ThreadImageRows(const Bounds& region, boost::function<void(Bounds)> function);
  // Split Bounds into num_threads chunks in y axis.
  void ThreadImageChunks(const Bounds& region, boost::function<void(Bounds)> function);
};

} // namespace afx

#endif  // THREADING_H_
