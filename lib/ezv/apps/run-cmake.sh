#!/usr/bin/env bash

rm -rf build

CC=${CC:-gcc} cmake -S . -B build -DEasypap_ROOT=/tmp/easypap/lib/cmake/Easypap -DCMAKE_BUILD_TYPE=Release || exit $?

cmake --build build || exit $?

