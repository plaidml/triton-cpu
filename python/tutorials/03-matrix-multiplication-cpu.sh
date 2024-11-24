#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

config=$1
shift
numthreads=$1
shift

block_pointers_via_raising=0

export BENCHMARK_BACKEND="triton-xsmm"

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
    --backend)
      shift
      export BENCHMARK_BACKEND=$1
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done


if [ "$config" = "baseline" ]; then
  if [ "$BENCHMARK_BACKEND" != "torch-cpu-native" ] && [ "$BENCHMARK_BACKEND" != "torch-cpu-compile" ]; then
    echo "ERROR: baseline config but backend is not torch-cpu-native or torch-cpu-compile"; exit 1
  fi
elif [ "$config" = "baseline-scalar" ]; then
  if [ "$BENCHMARK_BACKEND" != "triton-cpu" ]; then
    echo "ERROR: baseline-scalar config but backend is not triton-cpu"; exit 1
  fi
elif [ "$config" = "baseline-block" ]; then
  if [ "$BENCHMARK_BACKEND" != "triton-cpu" ]; then
    echo "ERROR: baseline-block config but backend is not triton-cpu"; exit 1
  fi
  export USE_BLOCK_POINTERS=1
elif [ "$config" = "xsmm-scalar" ]; then
  if [ "$BENCHMARK_BACKEND" != "triton-xsmm" ]; then
    echo "ERROR: xsmm config but backend is not triton-xsmm"; exit 1
  fi
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-block" ]; then
  if [ "$BENCHMARK_BACKEND" != "triton-xsmm" ]; then
    echo "ERROR: xsmm config but backend is not triton-xsmm"; exit 1
  fi
  if [ $block_pointers_via_raising = 1 ]; then
    export TRITON_CPU_RAISE_BLOCK_POINTER=1
  else
    export USE_BLOCK_POINTERS=1
  fi
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-pad-k" ]; then
  if [ "$BENCHMARK_BACKEND" != "triton-xsmm" ]; then
    echo "ERROR: xsmm config but backend is not triton-xsmm"; exit 1
  fi
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
  if [ "$BENCHMARK_BACKEND" != "triton-xsmm" ]; then
    echo "ERROR: xsmm config but backend is not triton-xsmm"; exit 1
  fi
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
if [ -e "$numthreads" ]; then
  echo "ERROR: must specify numthreads as 2nd arg"; exit 1
fi
export TRITON_CPU_MAX_THREADS=${numthreads}
export OMP_NUM_THREADS=${numthreads}

# Thread affinity changes with hyper-threading
THREADS_PER_CORE=$(lscpu | grep --color=never "Thread.*core" | tee - | grep -o "[0-9]\+")
SKIP=$((THREADS_PER_CORE-1)) # 0 for no HT, 1 for 2, 3 for 4, etc.
export KMP_AFFINITY=granularity=fine,compact,$SKIP,0

python $SCRIPT_DIR/03-matrix-multiplication-cpu.py
