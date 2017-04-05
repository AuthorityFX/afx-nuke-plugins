// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_THREADING_H_
#define AFX_THREADING_H_

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>

#include "include/bounds.h"

#ifdef _WIN32
  #ifdef _MSC_VER
    #pragma warning(disable: 4251)
  #endif
  #define WINLIB_EXPORT __declspec(dllexport)
#else
  #define WINLIB_EXPORT
#endif

namespace afx {

class WINLIB_EXPORT Threader {
 private:
  boost::asio::io_service io_service_;
  boost::thread_group thread_pool_;
  boost::scoped_ptr<boost::asio::io_service::work> work_ptr_;
  unsigned int num_threads_;

  boost::condition_variable exited_run_;
  boost::condition_variable io_service_ready_;
  bool running_;

  boost::mutex mutex_;

  // This function is pased to each thread in thread pool and will block for
  // work from queue added via AddWork()
  void Worker_();

 public:
   Threader();
   explicit Threader(unsigned int req_num_threads);
  ~Threader();
  // Launch additional threads.
  void AddThreads(unsigned int num_threads);
  // Start asio service. Launch requested number of threads. 0 will launch hardware concurency
  void InitializeThreads(unsigned int requested_threads = 0);
  // Block until all work has been completed.
  void Wait();
  // Synchonize and join all threads.
  void StopAndJoin();
  // Pass a function pointer to be executed by first available thread in pool
  void AddWork(boost::function<void()> function);
  // Can be used as an exit point
  bool IsRunning() const;
  unsigned int Threads() const;
};

class WINLIB_EXPORT ImageThreader : public Threader {
 public:
  // Split bounds into rows.
  void ThreadRows(const Bounds& region, boost::function<void(Bounds)> function);
  // Split bounds into columns.
  void ThreadColumns(const Bounds& region, boost::function<void(Bounds)> function);
  // Split Bounds into num_threads chunks in y axis.
  void ThreadRowChunks(const Bounds& region, boost::function<void(Bounds)> function);
  // Split Bounds into num_threads chunks in y axis.
  void ThreadColumnChunks(const Bounds& region, boost::function<void(Bounds)> function);
};

}  // namespace afx

#endif  // AFX_THREADING_H_
