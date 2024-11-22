#!/usr/bin/env bash

# saving path
HERE=$PWD

echo "===================================== Configuring Checkout"
git checkout dev-cpu-v2
git submodule init
git submodule update

echo "===================================== Setting Up Conda Env"
cd ./../
export CONDA_INSTALL_DIR=`pwd`/miniforge
export ENV_NAME=triton
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
bash ./Miniforge3-Linux-$(uname -m).sh -b -p ${CONDA_INSTALL_DIR}
${CONDA_INSTALL_DIR}/bin/conda create -y -n ${ENV_NAME} python=3.9
source ${CONDA_INSTALL_DIR}/bin/activate ${ENV_NAME}

echo "===================================== Install Dependencies"
pip install ninja cmake wheel pybind11 scipy numpy torch pytest lit pandas matplotlib
if [ $? != 0 ]; then
  exit 1
fi

echo "===================================== Building LLVM"
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout `cat ${HERE}/cmake/llvm-hash.txt`
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=True -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" ../llvm
ninja
cd ./../../

cd $HERE

conda deactivate
