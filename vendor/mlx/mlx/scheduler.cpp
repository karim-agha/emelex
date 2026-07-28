// Copyright © 2023 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/gpu/eval.h"

namespace mlx::core {

void synchronize(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    auto p = std::make_shared<std::promise<void>>();
    std::future<void> f = p->get_future();
    scheduler::enqueue_exception_barrier(s, [p = std::move(p), s]() {
      // emelex patch: inspect the stream's captured exception inside the
      // queued barrier. This makes the promise result describe exactly the
      // tasks before the barrier; a later task cannot race into this sync.
      try {
        scheduler::scheduler().rethrow_first_exception(s);
        p->set_value();
      } catch (...) {
        p->set_exception(std::current_exception());
      }
    });
    f.get();
  } else {
    gpu::synchronize(s);
  }
}

void synchronize(ThreadLocalStream s) {
  synchronize(stream_from_thread_local_stream(s));
}

void synchronize() {
  synchronize(default_stream(default_device()));
}

void clear_streams() {
  gpu::clear_streams();
}

namespace scheduler {

Scheduler::Scheduler() {
  gpu::init();
}

Scheduler::~Scheduler() = default;

void Scheduler::enqueue(Stream s, std::function<void()> task) {
  if (fail_next_enqueue_.exchange(false, std::memory_order_acq_rel)) {
    throw std::bad_alloc();
  }
  StreamThread* st = nullptr;
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      st = it->second.get();
    }
  }
  if (!st) {
    std::unique_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it == threads_.end()) {
      it = threads_.emplace(s.index, std::make_unique<StreamThread>()).first;
    }
    st = it->second.get();
  }
  st->enqueue(std::move(task));
}

void Scheduler::enqueue_exception_barrier(
    Stream s,
    std::function<void()> task) {
  StreamThread* st = nullptr;
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      st = it->second.get();
    }
  }
  if (!st) {
    std::unique_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it == threads_.end()) {
      it = threads_.emplace(s.index, std::make_unique<StreamThread>()).first;
    }
    st = it->second.get();
  }
  st->enqueue(std::move(task), true);
}

void Scheduler::rethrow_first_exception(Stream s) {
  StreamThread* st = nullptr;
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      st = it->second.get();
    }
  }
  if (st) {
    st->rethrow_first_exception();
  }
}

/** A singleton scheduler to manage devices, streams, and task execution. */
Scheduler& scheduler() {
  // Intentionally leaked to avoid the "static destruction order fiasco":
  // background threads (e.g. command buffer completion handlers) may
  // reference this singleton after other static objects are destroyed
  // during process teardown.
  static Scheduler* scheduler = new Scheduler;
  return *scheduler;
}

} // namespace scheduler
} // namespace mlx::core
