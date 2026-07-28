#!/bin/bash
#
# Generate C++ functions that expose Metal source to runtime JIT compilation.
#
# Copyright © 2023-25 Apple Inc.

set -eo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 OUTPUT_DIR XCRUN SOURCE_DIR SOURCE_FILE [METAL_FLAG ...]" >&2
  exit 64
fi

OUTPUT_DIR=$1
XCRUN=$2
SRC_DIR=$3
SRC_FILE=$4
shift 4
METAL_FLAGS=("$@")

SRC_NAME=$(basename -- "$SRC_FILE")
JIT_INCLUDES="${SRC_DIR}/mlx/backend/metal/kernels/jit"
INPUT_FILE="${SRC_DIR}/mlx/backend/metal/kernels/${SRC_FILE}.h"
OUTPUT_FILE="${OUTPUT_DIR}/${SRC_NAME}.cpp"

mkdir -p "$OUTPUT_DIR"

if ! HDRS=$(
  "$XCRUN" --sdk macosx metal -x metal \
    -I"$SRC_DIR" -I"$JIT_INCLUDES" -DMLX_METAL_JIT \
    -E -P -CC -C -H "$INPUT_FILE" "${METAL_FLAGS[@]}" -w \
    2>&1 1>/dev/null
); then
  echo "Metal compiler header resolution failed for ${INPUT_FILE}" >&2
  printf '%s\n' "$HDRS" >&2
  exit 1
fi

DEPTHS=()
HEADERS=()
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  case "$line" in
    .*" "*)
      marker=${line%% *}
      header=${line#* }
      ;;
    *)
      echo "Metal compiler returned an unexpected header line: ${line}" >&2
      exit 1
      ;;
  esac
  case "$marker" in
    *[!.]*)
      echo "Metal compiler returned an invalid header depth: ${line}" >&2
      exit 1
      ;;
  esac
  case "$header" in
    "$SRC_DIR"/*)
      DEPTHS+=("${#marker}")
      HEADERS+=("${header#"$SRC_DIR"/}")
      ;;
    *Xcode*)
      ;;
    *)
      echo "Metal compiler returned a header outside source tree: ${header}" >&2
      exit 1
      ;;
  esac
done <<EOF
$HDRS
EOF

STACK=()
SORTED=()
header_count=${#HEADERS[@]}
for ((index = 0; index < header_count; index += 1)); do
  depth_this=${DEPTHS[$index]}
  if [ $((index + 1)) -lt "$header_count" ]; then
    depth_next=${DEPTHS[$((index + 1))]}
  else
    depth_next=1
  fi
  header=${HEADERS[$index]}

  if [ "$depth_next" -gt "$depth_this" ]; then
    STACK=("$header" "${STACK[@]}")
    continue
  fi

  SORTED+=("$header")
  pop_count=$((depth_this - depth_next))
  if [ "$pop_count" -gt "${#STACK[@]}" ]; then
    echo "Metal compiler returned an invalid header-depth transition" >&2
    exit 1
  fi
  for ((stack_index = 0; stack_index < pop_count; stack_index += 1)); do
    SORTED+=("${STACK[$stack_index]}")
  done
  STACK=("${STACK[@]:$pop_count}")
done

if [ "${#STACK[@]}" -ne 0 ]; then
  echo "Metal compiler returned an unbalanced header tree" >&2
  exit 1
fi

SORTED+=("${INPUT_FILE#"$SRC_DIR"/}")

OUTPUT_TEMP="${OUTPUT_FILE}.tmp.$$"
trap 'rm -f "$OUTPUT_TEMP"' EXIT
{
  cat <<EOF
namespace mlx::core::metal {

const char* $SRC_NAME() {
  return R"preamble(
EOF
  echo "// Copyright © 2025 Apple Inc."
  echo
  echo "// Auto generated source for ${INPUT_FILE#"$SRC_DIR"/}"
  echo

  for header in "${SORTED[@]}"; do
    echo "///////////////////////////////////////////////////////////////////////////////"
    echo "// Contents from \"${header}\""
    echo "///////////////////////////////////////////////////////////////////////////////"
    echo
    echo "#line 1 \"${header}\""

    while IFS= read -r source_line || [ -n "$source_line" ]; do
      case "$source_line" in
        \#include\ \"*.h\"|\#pragma\ once)
          :
          ;;
        *)
          printf '%s\n' "$source_line"
          ;;
      esac
    done < "${SRC_DIR}/${header}"
    echo
  done

  echo "///////////////////////////////////////////////////////////////////////////////"
  cat <<EOF
)preamble";
}

} // namespace mlx::core::metal
EOF
} > "$OUTPUT_TEMP"
mv -f "$OUTPUT_TEMP" "$OUTPUT_FILE"
trap - EXIT
