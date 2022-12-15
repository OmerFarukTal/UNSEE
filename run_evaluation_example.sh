#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --partition=short        
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=2:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com

python3 evaluation.py \
    --model_name_or_path experiments/unsup-corinfomax-bert-base-uncased-b33 \
    --pooler cls_before_pooler \
    --proj_output_dim 4096-4096-128 \
    --task_set sts \
    --mode test