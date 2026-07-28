/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/metal.h"

#include <stdexcept>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_metal_is_available(bool* res) {
  try {
    *res = mlx::core::metal::is_available();
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_set_default_library_path(const char* path) {
  try {
    if (path == nullptr) {
      throw std::invalid_argument("default metallib path cannot be null");
    }
    mlx::core::metal::set_default_library_path(std::string(path));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_recommended_max_working_set_size(uint64_t* res) {
  try {
    if (res == nullptr) {
      throw std::invalid_argument("result pointer cannot be null");
    }
    auto pool = mlx::core::metal::new_scoped_memory_pool();
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) {
      throw std::runtime_error("no Metal device is available");
    }
    *res = static_cast<uint64_t>(device->recommendedMaxWorkingSetSize());
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_start_capture(const char* path) {
  try {
    mlx::core::metal::start_capture(std::string(path));
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_stop_capture(void) {
  try {
    mlx::core::metal::stop_capture();
  } catch (std::exception& e) {
    mlx_error("%s", e.what());
    return 1;
  }
  return 0;
}
