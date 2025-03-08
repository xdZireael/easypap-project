#!/usr/bin/env bash

EZP_INSTALL_DIR=/tmp/easypap

rm -rf build

CC=${CC:-gcc} cmake -S . -B build -DEasypap_ROOT=${EZP_INSTALL_DIR}/lib/cmake/Easypap -DCMAKE_BUILD_TYPE=Release || exit $?

cmake --build build --config Release || exit $?
