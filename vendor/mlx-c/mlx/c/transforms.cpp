/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include <atomic>
#include <stdexcept>
#include <string>

#include "mlx/c/transforms.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/transforms.h"

extern "C" int mlx_async_eval(const mlx_vector_array outputs) {
  try {
    mlx::core::async_eval(mlx_vector_array_get_(outputs));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_checkpoint(mlx_closure* res, const mlx_closure fun) {
  try {
    mlx_closure_set_(*res, mlx::core::checkpoint(mlx_closure_get_(fun)));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_custom_function(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp /* may be null */,
    const mlx_closure_custom_jvp fun_jvp /* may be null */,
    const mlx_closure_custom_vmap fun_vmap /* may be null */) {
  try {
    mlx_closure_set_(
        *res,
        mlx::core::custom_function(
            mlx_closure_get_(fun),
            (fun_vjp.ctx ? std::make_optional(mlx_closure_custom_get_(fun_vjp))
                         : std::nullopt),
            (fun_jvp.ctx
                 ? std::make_optional(mlx_closure_custom_jvp_get_(fun_jvp))
                 : std::nullopt),
            (fun_vmap.ctx
                 ? std::make_optional(mlx_closure_custom_vmap_get_(fun_vmap))
                 : std::nullopt)));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_custom_vjp(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp) {
  try {
    mlx_closure_set_(
        *res,
        mlx::core::custom_vjp(
            mlx_closure_get_(fun), mlx_closure_custom_get_(fun_vjp)));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_eval(const mlx_vector_array outputs) {
  try {
    mlx::core::eval(mlx_vector_array_get_(outputs));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_jvp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array tangents) {
  try {
    {
      auto [tpl_0, tpl_1] = mlx::core::jvp(
          mlx_closure_get_(fun),
          mlx_vector_array_get_(primals),
          mlx_vector_array_get_(tangents));
      mlx_vector_array_set_(*res_0, tpl_0);
      mlx_vector_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_value_and_grad(
    mlx_closure_value_and_grad* res,
    const mlx_closure fun,
    const int* argnums,
    size_t argnums_num) {
  try {
    mlx_closure_value_and_grad_set_(
        *res,
        mlx::core::value_and_grad(
            mlx_closure_get_(fun),
            std::vector<int>(argnums, argnums + argnums_num)));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_vjp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array cotangents) {
  try {
    {
      auto [tpl_0, tpl_1] = mlx::core::vjp(
          mlx_closure_get_(fun),
          mlx_vector_array_get_(primals),
          mlx_vector_array_get_(cotangents));
      mlx_vector_array_set_(*res_0, tpl_0);
      mlx_vector_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}

// emelex patch: private regression seam. Rust's subprocess test verifies an
// asynchronous CPU kernel exception crosses synchronize() and this C boundary
// as an ordinary mlx-c error instead of terminating the process.
extern "C" int mlx_emelex_test_cpu_scheduler_exception(void) {
  try {
    auto stream = mlx::core::new_stream(mlx::core::Device::cpu);
    auto& encoder = mlx::core::cpu::get_command_encoder(stream);
    for (int i = 0; i < mlx::core::cpu::DISPATCHES_PER_TASK - 1; ++i) {
      encoder.dispatch([] {});
    }
    encoder.dispatch([] {
      throw std::runtime_error("emelex CPU scheduler exception probe");
    });
    mlx::core::synchronize(stream);
    mlx_error("CPU scheduler swallowed the exception probe");
    return 1;
  } catch (std::exception& e) {
    if (mlx::core::scheduler::n_active_tasks() != 0) {
      mlx_error("CPU scheduler left grouped completion accounting active");
      return 1;
    }
    mlx_error("%s", e.what());
    return 1;
  }
}

// emelex patch: the scheduler must normalize a foreign/non-std throw before
// it reaches this std::exception-only generated C boundary.
extern "C" int mlx_emelex_test_cpu_scheduler_nonstandard_exception(void) {
  try {
    auto stream = mlx::core::new_stream(mlx::core::Device::cpu);
    auto& encoder = mlx::core::cpu::get_command_encoder(stream);
    for (int i = 0; i < mlx::core::cpu::DISPATCHES_PER_TASK - 1; ++i) {
      encoder.dispatch([] {});
    }
    encoder.dispatch([] { throw 42; });
    mlx::core::synchronize(stream);
    mlx_error("CPU scheduler swallowed the non-standard exception probe");
    return 1;
  } catch (std::exception& e) {
    if (mlx::core::scheduler::n_active_tasks() != 0) {
      mlx_error("CPU scheduler left non-standard probe accounting active");
      return 1;
    }
    mlx_error("%s", e.what());
    return 1;
  }
}

// emelex patch: a boundary enqueue failure must roll back both grouped-task
// registration and CommandEncoder grouping state.
extern "C" int mlx_emelex_test_cpu_enqueue_failure_recovery(void) {
  try {
    auto stream = mlx::core::new_stream(mlx::core::Device::cpu);
    auto& encoder = mlx::core::cpu::get_command_encoder(stream);
    for (int i = 0; i < mlx::core::cpu::DISPATCHES_PER_TASK - 1; ++i) {
      encoder.dispatch([] {});
    }
    mlx::core::scheduler::inject_enqueue_failure_for_test();
    try {
      encoder.dispatch([] {});
      mlx_error("CPU scheduler enqueue failure probe did not fail");
      return 1;
    } catch (const std::bad_alloc&) {
    }
    if (mlx::core::scheduler::n_active_tasks() != 0) {
      mlx_error("CPU enqueue failure left grouped completion accounting active");
      return 1;
    }

    std::atomic<int> completed{0};
    encoder.dispatch([&completed] { completed.fetch_add(1); });
    mlx::core::synchronize(stream);
    encoder.dispatch([&completed] { completed.fetch_add(1); });
    mlx::core::synchronize(stream);
    if (completed.load() != 2 ||
        mlx::core::scheduler::n_active_tasks() != 0) {
      mlx_error("CPU stream was not reusable after enqueue failure");
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
}

// emelex patch: a grouped completion wrapper skipped after an earlier stream
// exception must still release exactly one active-task registration.
extern "C" int mlx_emelex_test_cpu_skipped_group_completion(void) {
  try {
    auto stream = mlx::core::new_stream(mlx::core::Device::cpu);
    auto& encoder = mlx::core::cpu::get_command_encoder(stream);
    encoder.dispatch([] {
      throw std::runtime_error("emelex skipped group completion probe");
    });
    for (int i = 1; i < mlx::core::cpu::DISPATCHES_PER_TASK; ++i) {
      encoder.dispatch([] {});
    }

    try {
      mlx::core::synchronize(stream);
      mlx_error("CPU scheduler swallowed the skipped completion probe");
      return 1;
    } catch (const std::runtime_error& e) {
      if (std::string(e.what()).find("skipped group completion probe") ==
          std::string::npos) {
        throw;
      }
    }
    if (mlx::core::scheduler::n_active_tasks() != 0) {
      mlx_error("skipped CPU group left completion accounting active");
      return 1;
    }

    std::atomic<int> completed{0};
    encoder.dispatch([&completed] { completed.fetch_add(1); });
    mlx::core::synchronize(stream);
    if (completed.load() != 1 ||
        mlx::core::scheduler::n_active_tasks() != 0) {
      mlx_error("CPU stream was not reusable after skipped group");
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
}
