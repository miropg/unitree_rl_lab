#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=cdux
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --output=/projects/cdux/mirop/logs/test_gpu_%j.log

echo "Job started on $(hostname)"
echo "Checking GPU..."

singularity exec /projects/cdux/mirop/isaac-lab.sif nvidia-smi

echo "Checking Python..."
singularity exec /projects/cdux/mirop/isaac-lab.sif /isaac-sim/kit/python/bin/python3.12 --version

echo "Done!"