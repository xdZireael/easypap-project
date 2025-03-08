cmake_minimum_required(VERSION 3.18)
project(mesh LANGUAGES C)

# Ensure find_package prefers xxxConfig.cmake over Findxxx.cmake
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# Add the path of the custom module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0072 NEW) # to avoid prefer GLVND

# We need ezm (ezv will be included automatically)
find_package(Easypap COMPONENTS ezm REQUIRED)
find_package(OpenMP REQUIRED)

set(PSOURCES
    mesh.c
    mandel.c
)

foreach(src ${PSOURCES})
    get_filename_component(exec ${src} NAME_WE)
    add_executable(${exec} ${src})
    target_link_libraries(${exec} PRIVATE Easypap::ezm OpenMP::OpenMP_C)
    target_compile_options(${exec} PRIVATE "$<$<CONFIG:Release>:-march=native>")
endforeach()
