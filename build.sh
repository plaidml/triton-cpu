#!/usr/bin/env bash

source ./../miniforge/bin/activate triton

# Note, build and install LLVM on this directory with hash to avoid conflicts
export LLVM_BUILD_DIR=$PWD/../llvm-project/build
export TRITON_BUILD_WITH_CCACHE=false
export TRITON_BUILD_WITH_CLANG_LLD=true
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR

echo "===================================== Build"
pip install -e python/
if [ $? != 0 ]; then
  exit 1
fi

echo "===================================== CMake Tests"
ctest --test-dir python/build/cmake*
if [ $? != 0 ]; then
  exit 1
fi

echo "===================================== Setting up LIBXSMM for paddeing"
export XSMM_ROOT_DIR=$(realpath $(find python/build/ -type d -name xsmm-src | grep -v third_party))
export XSMM_LIB_DIR=$(realpath python/triton/_C)
cd third_party/cpu/python
python setup.py install
cd ../../../

conda deactivate
