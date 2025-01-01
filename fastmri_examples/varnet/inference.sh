#!/bin/bash

### GPU batch job ###
#SBATCH --job-name=varnet
#SBATCH --account=st-ilker-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/%x-%j_output.txt
#SBATCH --error=outputs/%x-%j_error.txt
# #SBATCH --mail-user=alvinbk@student.ubc.ca
# #SBATCH --mail-type=ALL

#############################################################################

SOCKEYE=1

accelerations=8
center_fractions=0.16
dataset=m4raw
defualt_root_dir=${dataset}/acc_${accelerations}_center_frac_${center_fractions}
output_path=${defualt_root_dir}
state_dict_file=${defualt_root_dir}/checkpoints
test_split=val

if [ $SOCKEYE -eq 1 ]; then
    target_path=~/project/EECE571/datasets/${dataset}/multicoil_${test_split}
    predictions_path=${defualt_root_dir}/reconstructions
    root_dir=~/project/EECE571/fastMRI/fastmri_examples/varnet

    source ~/project/EECE571/fastMRI/venv/bin/activate

    echo "Using sockeye"
    echo challenge: $challenge
    echo test_split: $test_split
    echo target_path: $target_path
    echo default_root_dir: $defualt_root_dir
    echo output_path: $output_path
    echo state_dict_file: $state_dict_file
    echo predictions_path: $predictions_path

else
    target_path=/home/alvin/UltrAi/Datasets/raw_datasets/${dataset}/multicoil_${test_split}
    predictions_path=${defualt_root_dir}/reconstructions
    root_dir="./"
    source venv/bin/activate
    export CUDA_VISIBLE_DEVICES=0
fi

python ${root_dir}/run_pretrained_varnet_inference.py \
    --state_dict_file $state_dict_file \
    --data_path $target_path \
    --output_path $output_path \
    --accelerations $accelerations \
    --center_fractions $center_fractions \