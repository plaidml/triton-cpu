#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

config=$1
shift
numthreads=$1
shift

block_pointers_via_raising=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --raise-block-pointers)
      shift
      block_pointers_via_raising=1
      ;;
    --external-pad)
      shift
      export PREPROCESS_EXTERNAL=1
      ;;
    --datatype)
      shift
      export DATATYPE=$1
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done


if [ "$config" = "xsmm-scalar" ]; then
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-block" ]; then
  if [ $block_pointers_via_raising = 1 ]; then
    export TRITON_CPU_RAISE_BLOCK_POINTER=1
  else
    export USE_BLOCK_POINTERS=1
  fi
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-pad-k" ]; then
  if [ $block_pointers_via_raising = 1 ]; then
    export TRITON_CPU_RAISE_BLOCK_POINTER=1
  else
    export USE_BLOCK_POINTERS=1
  fi
  export XSMM_PAD=1
  export K_DIM_PADDING=1
  export CACHE_PADDING=1
  export BLOCK_SIZE_K=512
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-loop-collapse-pad-b" ]; then
  if [ $block_pointers_via_raising = 1 ]; then
    export TRITON_CPU_RAISE_BLOCK_POINTER=1
  else
    export USE_BLOCK_POINTERS=1
  fi
  export XSMM_PAD=1
  export PAD_B_ONLY=1
  export BLOCK_SIZE_K=32
  export CACHE_PADDING=1
  export TRITON_CPU_LOOP_BRGEMM_XSMM=1
elif [ "$config" = "xsmm-external-pad" ]; then
  echo "NOT A TRUE CONFIG; try --external-pad on another config"
  exit 1
else
  echo "ERROR: unrecognized config: $config"
  exit 1
fi

# Uses the libxsmm built in the repo
export XSMM_LIB_DIR=$SCRIPT_DIR/../triton/_C/
export LD_LIBRARY_PATH=$XSMM_LIB_DIR:$LD_LIBRARY_PATH
export LD_PRELOAD=/lib64/libomp.so:$LD_PRELOAD
export TRITON_CPU_MAX_THREADS=${numthreads}
export OMP_NUM_THREADS=${numthreads}
# Hyper-Threading
export KMP_AFFINITY=granularity=fine,compact,1,0
# No Hyper-Threading
#export KMP_AFFINITY=granularity=core,compact,0,0

python $SCRIPT_DIR/03-matrix-multiplication-cpu.py

