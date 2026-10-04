#!/bin/sh
set -eu
cpp_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$cpp_dir/build"
binary="$cpp_dir/build/train"
if [ ! -x "$binary" ] || [ "$cpp_dir/trainer.cpp" -nt "$binary" ] || [ "$cpp_dir/poker.hpp" -nt "$binary" ]; then
    "${CXX:-c++}" -std=c++17 -O3 -DNDEBUG -pthread -Wall -Wextra -Wpedantic "$cpp_dir/trainer.cpp" -o "$binary.tmp"
    mv "$binary.tmp" "$binary"
fi
exec "$binary" "$@"
