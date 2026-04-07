#!/bin/bash
#PBS -N finetune-large
#PBS -l nodes=1:ppn=8:gpus=8
#PBS -l walltime=96:00:00
#PBS -q gpu
#PBS -o logs/finetune_$PBS_JOBID.out
#PBS -e logs/finetune_$PBS_JOBID.err

# Load required modules
module load cuda/12.1
module load cudnn/8.9
module load python/3.11
module load conda

# Activate virtual environment
source ~/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Get PBS node info
export PBS_NODEFILE=$PBS_NODEFILE
export NODE_COUNT=$(cat $PBS_NODEFILE | wc -l)

# Run distributed training
torchrun --nproc_per_node=8 --nnodes=$NODE_COUNT train.py \
    --model_name ${MODEL_NAME:-"meta-llama/Llama-2-70b-hf"} \
    --data_path ${DATA_PATH:-"./data/train.jsonl"} \
    --output_dir ./outputs/${PBS_JOBID} \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --max_seq_length 4096 \
    --fsdp