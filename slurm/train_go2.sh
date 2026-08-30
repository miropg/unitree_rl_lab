#!/bin/bash
#SBATCH --job-name=train_go2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=/projects/cdux/mirop/logs/train_go2_%j.log

echo "Job started on $(hostname)"
echo "Starting Go2 training..."

cd /projects/cdux/mirop/unitree_rl_lab

singularity exec --nv /projects/cdux/mirop/isaac-lab.sif /isaac-sim/python.sh scripts/rsl_rl/train.py --headless --task Unitree-Go2-Velocity

echo "Training done!"