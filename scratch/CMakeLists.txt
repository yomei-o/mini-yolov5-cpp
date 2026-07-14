cmake_minimum_required(VERSION 3.18)
project(scratch_yolo LANGUAGES CXX)

# From-scratch YOLO. The learning core (autograd / conv / BatchNorm / CIoU /
# training) is STANDARD LIBRARY ONLY — no LibTorch, no BLAS, no Eigen.
# The only vendored code is stb_image / stb_image_write (single-header,
# public-domain), used solely to decode/encode JPG/PNG for convenience; the
# pure-C++ PPM path still works with zero dependencies.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Non-recursive on purpose: keep generated files under build/ out of the glob.
file(GLOB SCRATCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/*.cpp")
add_executable(scratch_yolo ${SCRATCH_SOURCES})
target_include_directories(scratch_yolo PRIVATE "${CMAKE_SOURCE_DIR}")

# /arch:AVX2 and -march=native are pure compiler flags (no external dependency);
# they let the compiler auto-vectorize the GEMM inner loops.
if(MSVC)
  target_compile_options(scratch_yolo PRIVATE /O2 /EHsc /utf-8 /arch:AVX2 /fp:fast)
else()
  target_compile_options(scratch_yolo PRIVATE -O2 -march=native -ffast-math)
endif()
