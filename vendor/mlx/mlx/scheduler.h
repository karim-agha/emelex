// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "mlx/api.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace mlx::core::scheduler {

struct StreamThread {
  struct QueuedTask {
    std::function<void()> task;
    bool exception_barrier;
  };

  std::mutex mtx;
  std::queue<QueuedTask> q;
  // emelex patch: worker exceptions cross the next caller barrier instead of
  // unwinding out of std::thread and terminating the process.
  std::exception_ptr first_exception;
  std::condition_variable cond;
  bool stop;
  std::thread thread;

  StreamThread() : stop(false), thread(&StreamThread::thread_fn, this) {}

  ~StreamThread() {
    {
      std::lock_guard<std::mutex> lk(mtx);
      stop = true;
    }
    cond.notify_one();
    thread.join();
  }

  void thread_fn() {
    while (true) {
      std::function<void()> task;
      bool skip_task = false;
      {
        std::unique_lock<std::mutex> lk(mtx);
        cond.wait(lk, [this] { return !this->q.empty() || this->stop; });
        if (q.empty() && stop) {
          return;
        }
        auto queued = std::move(q.front());
        q.pop();
        task = std::move(queued.task);
        // emelex patch: once one task fails, dependent work may observe
        // incomplete outputs. Drain it without execution until the explicit
        // synchronize barrier reports and clears the exception.
        skip_task = first_exception && !queued.exception_barrier;
      }
      if (skip_task) {
        continue;
      }

      // emelex patch: no exception may escape the scheduler thread. mlx-c
      // catches std::exception, so normalize foreign/non-standard throws
      // before retaining them for caller-thread rethrow.
      std::exception_ptr task_exception;
      try {
        task();
      } catch (const std::exception&) {
        task_exception = std::current_exception();
      } catch (...) {
        task_exception = std::make_exception_ptr(
            std::runtime_error("non-standard C++ exception in MLX task"));
      }
      if (task_exception) {
        std::lock_guard<std::mutex> lk(mtx);
        if (!first_exception) {
          first_exception = std::move(task_exception);
        }
      }
    }
  }

  void enqueue(std::function<void()> f, bool exception_barrier = false) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      if (stop) {
        throw std::runtime_error(
            "Cannot enqueue work after stream is stopped.");
      }
      q.emplace(QueuedTask{std::move(f), exception_barrier});
    }
    cond.notify_one();
  }

  void rethrow_first_exception() {
    std::exception_ptr exception;
    {
      std::lock_guard<std::mutex> lk(mtx);
      exception = std::exchange(first_exception, nullptr);
    }
    if (exception) {
      std::rethrow_exception(exception);
    }
  }
};

class MLX_API Scheduler {
 public:
  Scheduler();
  ~Scheduler();

  // Not copyable or moveable
  Scheduler(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  void enqueue(Stream s, std::function<void()> task);
  void enqueue_exception_barrier(Stream s, std::function<void()> task);
  void rethrow_first_exception(Stream s);
  void inject_enqueue_failure_for_test() {
    fail_next_enqueue_.store(true, std::memory_order_release);
  }

  void notify_new_task(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_++;
    }
    completion_cv.notify_all();
  }

  void notify_task_completion(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_--;
    }
    completion_cv.notify_all();
  }

  int n_active_tasks() const {
    // emelex patch: transforms inspect this outside `mtx`; an atomic load
    // avoids the upstream data race while writers retain the condition-
    // variable mutex needed to prevent lost wakeups.
    return n_active_tasks_.load(std::memory_order_acquire);
  }

  void wait_for_one() {
    std::unique_lock<std::mutex> lk(mtx);
    int n_tasks_old = n_active_tasks();
    if (n_tasks_old > 1) {
      completion_cv.wait(lk, [this, n_tasks_old] {
        return this->n_active_tasks() < n_tasks_old;
      });
    }
  }

 private:
  friend Stream mlx::core::new_stream(Device d);

  std::atomic<int> n_active_tasks_{0};
  std::atomic<bool> fail_next_enqueue_{false};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
  std::condition_variable completion_cv;
  std::mutex mtx;
};

MLX_API Scheduler& scheduler();

template <typename F>
void enqueue(const Stream& stream, F&& f) {
  scheduler().enqueue(stream, std::forward<F>(f));
}

template <typename F>
void enqueue_exception_barrier(const Stream& stream, F&& f) {
  scheduler().enqueue_exception_barrier(stream, std::forward<F>(f));
}

inline int n_active_tasks() {
  return scheduler().n_active_tasks();
}

inline void notify_new_task(const Stream& stream) {
  scheduler().notify_new_task(stream);
}

inline void notify_task_completion(const Stream& stream) {
  scheduler().notify_task_completion(stream);
}

inline void wait_for_one() {
  scheduler().wait_for_one();
}

inline void inject_enqueue_failure_for_test() {
  scheduler().inject_enqueue_failure_for_test();
}

} // namespace mlx::core::scheduler
