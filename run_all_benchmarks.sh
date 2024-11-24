#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Default data type
DATATYPE=f32
# Default to thread 0 on every core of the first socket
# _see KMP_AFFINITY in 03-matrix-multiplication-cpu.sh_
THREADS=$(lscpu | grep --color=never "Core.*socket" | grep -o "[0-9]\+")
# Default to run XSMM benchmarks
BASELINE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    bf16)
      DATATYPE=$1
      shift
      ;;
    bf8)
      DATATYPE=$1
      shift
      ;;
    --baseline)
      BASELINE=1
      shift
      ;;
    *)
      if ! [[ "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Syntax: run_all_benchmarks <f32|bf16|bf8> <THREADS> [--baseline]"
        exit 1
      fi
      THREADS=$1
      shift
      ;;
  esac
done
echo "DATATYPE = $DATATYPE"
echo "Num Threads = $THREADS"

source ../miniforge/bin/activate triton

# Debug only
#RUN=echo
RUN=time

# Single-threaded just needs to run once for Torch baseline, because every multi-threaded run also runs single threaded
echo -e "\nSingle-Threaded Baseline"
$RUN $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh xsmm-scalar $THREADS --datatype $DATATYPE

# Multi-threaded has two "flavours": baseline and XSMM
echo -e "\nMulti-Threaded Baseline"
for benchmark in xsmm-scalar xsmm-block; do
  echo -e "\nRUN: $benchmark | threads $THREADS | type $DATATYPE"
  $RUN $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $benchmark $THREADS --datatype $DATATYPE $external_pad
done

# This should only run on a branch where the XSMM bridge is properly installed
if [ $BASELINE == 0 ]; then
  echo -e "\nMulti-Threaded XSMM"
  for benchmark in xsmm-pad-k xsmm-loop-collapse-pad-b; do
    for external_pad in "" "--external-pad"; do
      echo -e "\nRUN: $benchmark | threads $THREADS | type $DATATYPE | $external_pad"
      $RUN $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $benchmark $THREADS --datatype $DATATYPE $external_pad
    done
  done
fi

conda deactivate
