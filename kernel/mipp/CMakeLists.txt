file(GLOB MIPP_SOURCES *.cpp)

target_sources(easypap PRIVATE ${MIPP_SOURCES})
target_include_directories(easypap PRIVATE ${CMAKE_SOURCE_DIR}/lib/mipp/include)
