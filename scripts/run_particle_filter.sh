#!/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_particle_filter
#SBATCH --time=01:00:00
#SBATCH --mem=10000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.cooper@monash.edu
#SBATCH --output=run-particle-filter-%j.out

module load cuda/8.0
module load anaconda/5.1.0-Python3.6-gcc5
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/lib/cuda/lib64:/usr/local/python/3.5.1-gcc/lib
# export PYTHONPATH=~/venv/lib/python3.5/site-packages:~/.local/lib/python3.5/site-packages

source activate pytorch
cd ~/src/ptvi/scripts

python run_particle_filter.py

# venv/bin/python tf-dummy.py

