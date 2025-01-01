#!/bin/bash

SOCKEYE=1

accelerations=4
center_fractions=0.08
challenge=multicoil
dataset=m4raw
defualt_root_dir=${dataset}/acc_${accelerations}_center_frac_${center_fractions}
predictions_path=${defualt_root_dir}/reconstructions
test_split=val

if [ $SOCKEYE -eq 1 ]; then
    target_path=~/project/EECE571/datasets/${dataset}/multicoil_${test_split}
    root_dir=~/project/EECE571/fastMRI/fastmri_examples/varnet

    source ~/project/EECE571/fastMRI/venv/bin/activate

    echo "Using sockeye"
    echo challenge: $challenge
    echo test_split: $test_split
    echo target_path: $target_path
    echo default_root_dir: $defualt_root_dir

else
    target_path=/home/alvin/UltrAi/Datasets/raw_datasets/${dataset}/multicoil_${test_split}
    root_dir="./"
    source venv/bin/activate
    export CUDA_VISIBLE_DEVICES=0
fi


python ${root_dir}/evaluate.py \
    --target-path $target_path \
    --predictions-path $predictions_path \
    --challenge $challenge