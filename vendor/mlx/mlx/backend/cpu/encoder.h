// Copyright © 2025 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/array.h"
#include "mlx/scheduler.h"

namespace mlx::core::cpu {

// Number of dispatches per scheduler task
constexpr int DISPATCHES_PER_TASK = 10;

// emelex patch: keep grouped-task accounting balanced when a dispatched CPU
// kernel throws. The scheduler worker catches and records the exception.
class GroupedTaskCompletion {
 public:
  explicit GroupedTaskCompletion(Stream stream) : stream_(stream) {
    scheduler::notify_new_task(stream_);
  }
  ~GroupedTaskCompletion() noexcept {
    // Never let scheduler bookkeeping unwind through a worker exception
    // boundary. std::mutex acquisition is the only operation here that may
    // theoretically throw; task execution must still remain contained.
    try {
      scheduler::notify_task_completion(stream_);
    } catch (...) {
    }
  }

  GroupedTaskCompletion(const GroupedTaskCompletion&) = delete;
  GroupedTaskCompletion& operator=(const GroupedTaskCompletion&) = delete;

 private:
  Stream stream_;
};

struct MLX_API CommandEncoder {
  CommandEncoder(Stream stream) : stream_(stream) {}

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;
  CommandEncoder(CommandEncoder&&) = delete;
  CommandEncoder& operator=(CommandEncoder&&) = delete;

  void set_input_array(const array& a) {}
  void set_output_array(array& a) {}

  // Hold onto a temporary until any already scheduled tasks which use it as
  // an input are complete.
  void add_temporary(array arr) {
    temporaries_.push_back(std::move(arr));
  }

  void add_temporaries(std::vector<array> arrays) {
    temporaries_.insert(
        temporaries_.end(),
        std::make_move_iterator(arrays.begin()),
        std::make_move_iterator(arrays.end()));
  }

  std::vector<array>& temporaries() {
    return temporaries_;
  }

  template <class F, class... Args>
  void dispatch(F&& f, Args&&... args) {
    const int previous_num_ops = num_ops_;
    num_ops_ = (num_ops_ + 1) % DISPATCHES_PER_TASK;
    try {
      auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
      if (num_ops_ == 0) {
        auto completion = std::make_shared<GroupedTaskCompletion>(stream_);
        // Keep one producer-side owner until enqueue returns. On enqueue
        // failure the queued callable may already have been destroyed; on a
        // worker exception it may be quarantined without execution. Either
        // way, the last shared owner balances exactly one registration.
        auto task_wrap = [completion, task = std::move(task)]() mutable {
          task();
        };
        scheduler::enqueue(stream_, std::move(task_wrap));
      } else {
        scheduler::enqueue(stream_, std::move(task));
      }
    } catch (...) {
      // Dispatch has no queue ownership on failure. Restore grouping state so
      // the next call retries the same boundary instead of losing accounting.
      num_ops_ = previous_num_ops;
      throw;
    }
  }

 private:
  Stream stream_;
  std::vector<array> temporaries_;
  int num_ops_{0};
};

MLX_API CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::cpu
