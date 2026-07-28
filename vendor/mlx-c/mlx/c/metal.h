/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* Generated upstream, then patched by Emelex. See vendor/PINS.md. */
/*                                                    */

#ifndef MLX_METAL_H
#define MLX_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup metal Metal specific operations
 */
/**@{*/

int mlx_metal_is_available(bool* res);
int mlx_metal_set_default_library_path(const char* path);
int mlx_metal_recommended_max_working_set_size(uint64_t* res);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture(void);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
