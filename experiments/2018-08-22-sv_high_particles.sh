#!/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_particle_filter_high_particles
#SBATCH --time=23:00:00
#SBATCH --mem=16000
#SBATCH --array=1-10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.cooper@monash.edu
#SBATCH --output=run-particle-filter-high-particles-%j.out

echo Large number of particles

module load cuda/8.0
module load anaconda/5.1.0-Python3.6-gcc5

source activate pytorch
cd ~/src/ptvi/experiments

mkdir T200P10k
cd T200P10k

sim-particle-filter 200 --gpu --fileprefix="high_particles_${SLURM_ARRAY_TASK_ID}" --algoseed ${SLURM_ARRAY_TASK_ID} --particles 10000
