#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ../miniforge/bin/activate triton

for datatype in f32 bf16; do
  for num_threads in 1 $(nproc); do
    for benchmark in xsmm-scalar xsmm-block; do
      echo -e "\n\nRUN: $benchmark | threads $num_threads | type $datatype"
      time $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $benchmark $num_threads --datatype $datatype $external_pad
    done
    for benchmark in xsmm-pad-k xsmm-loop-collapse-pad-b; do
      for external_pad in "" "--external-pad"; do
        echo -e "\n\nRUN: $benchmark | threads $num_threads | type $datatype | $external_pad"
        time $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $benchmark $num_threads --datatype $datatype $external_pad
      done
    done
  done
done

conda deactivate
