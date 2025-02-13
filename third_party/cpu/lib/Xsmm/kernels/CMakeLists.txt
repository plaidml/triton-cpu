find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Simple utilities to help exploring what static kernels are available.
add_triton_library(TritonCPUXsmmKernelUtils
  KernelUtils.cpp
)

# Hack to get python dependencies.
# Somehow python execution in this context is isolated from user local modules.
# Thus, a separate virtual env is created to install dependencies for static lib script.
add_custom_command(
    OUTPUT xsmm-venv
    COMMAND ${Python3_EXECUTABLE} -m venv xsmm-venv
)
add_custom_command(
    OUTPUT xsmm-venv.stamp
    DEPENDS xsmm-venv requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/xsmm-venv/bin/pip install -r ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt --upgrade
)
# Generate static LIBXSMM kernels.
add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels/XsmmKernels.h
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels/XsmmKernels.a

  DEPENDS
  xsmm
  xsmm-venv.stamp
  xsmm-generator
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel-config.json

  # Hack to make static lib script work.
  # The generator script has assumptions on relative location of its input files.
  # TODO: Generalize staticlib script to not rely on relative paths for binaries
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libxsmm_staticlib.py ${CMAKE_CURRENT_BINARY_DIR}/bin/libxsmm_staticlib.py
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kernel-config.json ${CMAKE_CURRENT_BINARY_DIR}/kernel-config.json
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/third_party/cpu/xsmm-generator ${CMAKE_CURRENT_BINARY_DIR}/bin/libxsmm_binaryexport_generator
  # Generate static kernel objects.
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/xsmm-venv/bin/python ${CMAKE_CURRENT_BINARY_DIR}/bin/libxsmm_staticlib.py -c ${CMAKE_CURRENT_BINARY_DIR}/kernel-config.json -o ${CMAKE_CURRENT_BINARY_DIR}
)
# CMake target to trigger custom object generation.
add_custom_target(TritonCPUXsmmGenKernels
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels/XsmmKernels.h
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels/XsmmKernels.a
)

add_triton_library(TritonCPUXsmmKernels
  MlirKernelWrappers.cpp

  DEPENDS
  TritonCPUXsmmGenKernels

  LINK_LIBS PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels/XsmmKernels.a
)
target_include_directories(TritonCPUXsmmKernels
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels
)
set_property(TARGET TritonCPUXsmmKernels PROPERTY CXX_STANDARD 11)
target_compile_definitions(TritonCPUXsmmKernels PRIVATE mlir_c_runner_utils_EXPORTS)
