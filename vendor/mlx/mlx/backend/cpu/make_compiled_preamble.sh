#!/bin/bash
#
# Generate the CPU source preamble used by runtime kernel compilation.
#
# Copyright © 2023-24 Apple Inc.

set -eo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 OUTPUT_FILE COMPILER SOURCE_DIR IS_CLANG ARCH SDK_ROOT" >&2
  exit 64
fi

OUTPUT_FILE=$1
COMPILER=$2
SOURCE_DIR=$3
IS_CLANG=$4
ARCH=$5
SDK_ROOT=$6
INCLUDES=

if [ "$IS_CLANG" = "TRUE" ]; then
  if [ ! -d "$SDK_ROOT" ]; then
    echo "CPU preamble SDK root is absent: ${SDK_ROOT}" >&2
    exit 1
  fi
  COMPILER_FLAGS=(
    -x c++
    -std=c++17
    -arch "$ARCH"
    -isysroot "$SDK_ROOT"
    -fkeep-system-includes
    -Wno-pragma-once-outside-header
  )
else
  COMPILER_FLAGS=(-std=c++17)
fi

if ! CONTENT=$(
  "$COMPILER" "${COMPILER_FLAGS[@]}" -I "$SOURCE_DIR" -E -P \
    "$SOURCE_DIR/mlx/backend/cpu/compiled_preamble.h"
); then
  echo "CPU preamble compiler failed for ${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$OUTPUT_FILE")"
OUTPUT_TEMP="${OUTPUT_FILE}.tmp.$$"
trap 'rm -f "$OUTPUT_TEMP"' EXIT
cat > "$OUTPUT_TEMP" <<EOF
const char* get_kernel_preamble() {
return R"preamble(
$INCLUDES
$CONTENT
using namespace mlx::core;
using namespace mlx::core::detail;
)preamble";
}
EOF
mv -f "$OUTPUT_TEMP" "$OUTPUT_FILE"
trap - EXIT
