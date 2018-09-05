#!/bin/env bash
#SBATCH --partition=short
#SBATCH --job-name=sv_mcmc_vs_vi
#SBATCH --time=23:00:00
#SBATCH --mem=16000
#SBATCH --array=100,200,500,1000,2000,5000,10000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.cooper@monash.edu
#SBATCH --output=sv_mcmc_vs_vi-%j.out

echo Stoch Vol MCMC vs VI

module load cuda/8.0
module load anaconda/5.1.0-Python3.6-gcc5

source activate pytorch
cd ~/src/ptvi/experiments

cd SV_MCMC_vs_VI

# we will assume the data file experiment.csv with 1e4 rows already exists

stochvol --data_seed=123 --algo_seed=123 conditional experiment.csv ${SLURM_ARRAY_TASK_ID} "VI${SLURM_ARRAY_TASK_ID}.json" --N=500 --a=1. --b=0. --c=0.8
stochvol --algo_seed=123 mcmc "VI${SLURM_ARRAY_TASK_ID}.json" "MCMC${SLURM_ARRAY_TASK_ID}.json"
stochvol compare "VI${SLURM_ARRAY_TASK_ID}.json" "MCMC${SLURM_ARRAY_TASK_ID}.json" "Forecast_${SLURM_ARRAY_TASK_ID}.pdf"
