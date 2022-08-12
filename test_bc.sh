#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=graphs_12q_test
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=0-1
#SBATCH --partition=test

module load languages/anaconda3/2021.09-3.9.7-tflow.1.12-deeplabcut
date
python3 gen_12_data_batches.py -arix=${SLURM_ARRAY_TASK_ID} -p="$PWD/data/uib_data" -od="$PWD/graph_data_12q" -df="entanglement12.bz2" -bs=2
date
