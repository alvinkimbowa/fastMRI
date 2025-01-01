#!/bin/bash

### GPU batch job ###
#SBATCH --job-name=varnet
#SBATCH --account=st-ilker-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=5-10:00:00
#SBATCH --output=outputs/%x-%j_output.txt
#SBATCH --error=outputs/%x-%j_error.txt
# #SBATCH --mail-user=alvinbk@student.ubc.ca
# #SBATCH --mail-type=ALL

#############################################################################

SOCKEYE=1

accelerations=8
center_fractions=0.04
challenge=multicoil
dataset=m4raw
defualt_root_dir=${dataset}_full/acc_${accelerations}_center_frac_${center_fractions}
num_workers=4
batch_size=1
num_gpus=1
sample_rate=1
val_sample_rate=1
test_split=val
test_volume_sample_rate=1
mode=train
if [ $mode == "test" ]; then
    batch_size=1
fi

if [ $SOCKEYE -eq 1 ]; then
    data_path=~/project/EECE571/datasets/${dataset}
    source ~/project/EECE571/fastMRI/venv/bin/activate
    root_dir=~/project/EECE571/fastMRI/fastmri_examples/varnet

    echo "Using sockeye"
    echo accelarations: $accelerations
    echo center_fractions: $center_fractions
    echo challenge: $challenge
    echo default_root_dir: $defualt_root_dir
    echo batch_size: $batch_size
    echo num_gpus: $num_gpus
    echo num_workers: $num_workers
    echo sample_rate: $sample_rate
    echo val_sample_rate: $val_sample_rate
    echo test_split: $test_split
    echo test_volume_sample_rate: $test_volume_sample_rate
    echo data_path: $data_path
    echo mode: $mode

else
    data_path=/home/alvin/UltrAi/Datasets/raw_datasets/${dataset}/
    source venv/bin/activate
    root_dir="./"
    export CUDA_VISIBLE_DEVICES=0
fi


python ${root_dir}/train_varnet_demo.py \
    --challenge $challenge \
    --data_path $data_path \
    --default_root_dir $defualt_root_dir \
    --accelerations $accelerations \
    --center_fractions $center_fractions \
    --batch_size $batch_size \
    --gpus $num_gpus \
    --num_workers $num_workers \
    --sample_rate $sample_rate \
    --val_sample_rate $val_sample_rate \
    --test_split $test_split \
    --test_volume_sample_rate $test_volume_sample_rate \
    --mode $mode