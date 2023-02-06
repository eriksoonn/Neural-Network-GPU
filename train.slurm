#!/bin/bash

# Job name
#SBATCH --job-name=train-NN

# Define the files which will contain the Standard and Error output
#SBATCH --output=outputs/M_%A.out
#SBATCH --error=outputs/M_%A.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use
# #SBATCH --cpus-per-task=10
# #SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=1

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=2G

# Select one partition
#SBATCH --partition=CTOS

#cmd

/home/erik/EHP/CUDA/Praktika/Neural-Network-GPU/bin/nn --train --learning_rate 0.01 --epochs 10000 --batch_number 10 --dataset /home/erik/EHP/CUDA/Praktika/datasets/CC_train.csv --layers 30,60,10,1 -s 1 --model /home/erik/EHP/CUDA/Praktika/Neural-Network-GPU/model.m --verbose