// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

// emelex patch: one-shot ownership for scheduler accounting. Construction
// increments before callback registration; any registration/commit failure
// rolls back through RAII, while a successful callback completes exactly once.
class ActiveTaskCompletion {
 public:
  explicit ActiveTaskCompletion(Stream stream) : stream_(stream) {
    scheduler::notify_new_task(stream_);
  }

  ~ActiveTaskCompletion() noexcept {
    complete();
  }

  ActiveTaskCompletion(const ActiveTaskCompletion&) = delete;
  ActiveTaskCompletion& operator=(const ActiveTaskCompletion&) = delete;

  void complete() noexcept {
    if (!active_.exchange(false, std::memory_order_acq_rel)) {
      return;
    }
    try {
      scheduler::notify_task_completion(stream_);
    } catch (...) {
      // Never unwind through Metal's completion thread or a destructor.
    }
  }

 private:
  Stream stream_;
  std::atomic<bool> active_{true};
};

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    auto completion = std::make_shared<ActiveTaskCompletion>(s);
    command_buffer->addCompletedHandler(
        [completion, buffers = std::move(buffers)](
            MTL::CommandBuffer*) noexcept {
          completion->complete();
        });
    try {
      encoder.commit();
    } catch (...) {
      completion->complete();
      throw;
    }
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer*) noexcept {});
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  encoder.end_encoding();
  encoder.commit();
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize_and_rethrow();
}

void clear_streams() {
  metal::get_command_encoders().clear();
}

} // namespace mlx::core::gpu
