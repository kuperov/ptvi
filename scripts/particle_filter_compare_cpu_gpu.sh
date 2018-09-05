#!/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_particle_filter_compare_gpu
#SBATCH --time=23:00:00
#SBATCH --mem=16000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.cooper@monash.edu
#SBATCH --output=run-particle-filter-compare-gpu-%j.out

echo Job for examining relative performance of CPU and GPU jobs

module load cuda/8.0
module load anaconda/5.1.0-Python3.6-gcc5

source activate pytorch
cd ~/src/ptvi/experiments

mkdir -f T200P1000
cd T200P1000

sim-particle-filter 200 --cpu --fileprefix='cpu1' --algoseed 1 --particles 1000
sim-particle-filter 200 --cpu --fileprefix='cpu2' --algoseed 2 --particles 1000
sim-particle-filter 200 --cpu --fileprefix='cpu3' --algoseed 3 --particles 1000

sim-particle-filter 200 --gpu --fileprefix='gpu1' --algoseed 1 --particles 1000
sim-particle-filter 200 --gpu --fileprefix='gpu2' --algoseed 2 --particles 1000
sim-particle-filter 200 --gpu --fileprefix='gpu3' --algoseed 3 --particles 1000
