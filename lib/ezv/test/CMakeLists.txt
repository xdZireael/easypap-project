cmake_minimum_required(VERSION 3.18)
project(examples LANGUAGES C)


# Ensure find_package prefers xxxConfig.cmake over Findxxx.cmake
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# Add the path of the custom module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0072 NEW) # to avoid prefer GLVND

# We only need ezv from EasyPap
find_package(Easypap COMPONENTS ezv REQUIRED)

set(PSOURCES
    boolmat.c       diy.c           imagecpu.c      mesh.c          morton.c        part.c          store.c
    cpu.c           image.c         imagedapu.c     monitor.c       paint.c         patch.c         thr.c
)

foreach(src ${PSOURCES})
    get_filename_component(exec ${src} NAME_WE)
    add_executable(${exec} ${src})
    target_link_libraries(${exec} Easypap::ezv m)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${exec} PRIVATE -march=native)
    endif()
endforeach()
