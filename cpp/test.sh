#!/bin/sh
set -eu
cpp_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$cpp_dir/build"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Wpedantic "$cpp_dir/test.cpp" -o "$cpp_dir/build/test"
"$cpp_dir/build/test"
