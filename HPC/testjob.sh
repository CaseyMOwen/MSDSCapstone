#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name=test-job
#SBATCH --time=00-00:20:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --mem=2g 
#SBATCH --output=MyJob.%j.%N.out
#SBATCH --error=MyJob.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=casey.owen@tufts.edu


module load anaconda/2021.05
python testscript.py 
