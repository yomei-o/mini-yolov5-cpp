cmake_minimum_required(VERSION 3.18)
project(yolov5_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---------------------------------------------------------------
# third_party lives one level up (C:/prog/vc/third_party) so it stays OUT of
# the project zip. Override with -DCMAKE_PREFIX_PATH=... if it lives elsewhere.
if(NOT DEFINED CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../third_party/libtorch")
endif()
find_package(Torch REQUIRED)

# OpenCV is optional: used for image IO / augmentation / visualization.
# If not found we fall back to a minimal stb-based image loader.
find_package(OpenCV QUIET)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(yolov5 ${SOURCES})
target_include_directories(yolov5 PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/../third_party/stb")
target_link_libraries(yolov5 "${TORCH_LIBRARIES}")

if(OpenCV_FOUND)
  message(STATUS "OpenCV found: ${OpenCV_VERSION} -> image IO via OpenCV")
  target_compile_definitions(yolov5 PRIVATE HAVE_OPENCV=1)
  target_include_directories(yolov5 PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(yolov5 ${OpenCV_LIBS})
else()
  message(STATUS "OpenCV NOT found -> using bundled stb_image loader")
endif()

if(MSVC)
  target_compile_options(yolov5 PRIVATE /EHsc /bigobj /utf-8)
endif()

# On Windows the LibTorch .dll files must sit next to the .exe at runtime.
if(MSVC)
  file(GLOB TORCH_DLLS "${CMAKE_PREFIX_PATH}/lib/*.dll")
  add_custom_command(TARGET yolov5 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:yolov5>)
endif()
