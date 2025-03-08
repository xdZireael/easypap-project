#!/usr/bin/env bash

rm -rf build

# Configure
CC=${CC:-gcc} CXX=${CXX:-g++} cmake -S . -B build -DCMAKE_BUILD_TYPE=Release || exit $?

# Build all
cmake --build build --parallel || exit $?
