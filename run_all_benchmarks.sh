#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ../miniforge/bin/activate triton

THREADS=$(lscpu | grep --color=never "Core.*socket" | grep -o "[0-9]\+")

commit=$(git describe --always --dirty)

for datatype in f32 bf16 bf8; do
  for num_threads in 1 $THREADS; do
    for backend in torch-cpu-compile torch-cpu-native; do
      config=baseline
      echo -e "BENCHMARK: {'backend': '$backend', 'config': '$config', 'threads': $num_threads, 'type': '$datatype', 'commit': '$commit'}"
      time $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $config $num_threads --datatype $datatype --backend $backend 2>&1
      echo -e "\n\n"
    done

    backend=triton-cpu
    for config in baseline-scalar baseline-block; do
      echo -e "BENCHMARK: {'backend': '$backend', 'config': '$config', 'threads': $num_threads, 'type': '$datatype', 'commit': '$commit'}"
      time $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $config $num_threads --datatype $datatype --backend $backend 2>&1
      echo -e "\n\n"
    done

    # Triton-XSMM
    backend=triton-xsmm
    for config in xsmm-scalar xsmm-block; do
      echo -e "BENCHMARK: {'backend': '$backend', 'config': '$config', 'threads': $num_threads, 'type': '$datatype', 'commit': '$commit'}"
      time $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $config $num_threads --datatype $datatype --backend $backend 2>&1
      echo -e "\n\n"
    done
    for config in xsmm-pad-k xsmm-loop-collapse-pad-b; do
      for external_pad in "" "--external-pad"; do
        echo -e "BENCHMARK: {'backend': '$backend', 'config': '$config', 'threads': $num_threads, 'type': '$datatype', 'pad': '$external_pad', 'commit': '$commit'}"
        time $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.sh $config $num_threads --datatype $datatype --backend $backend $external_pad 2>&1
        echo -e "\n\n"
      done
    done
  done
done

conda deactivate
