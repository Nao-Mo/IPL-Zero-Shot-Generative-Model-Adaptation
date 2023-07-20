#! /bin/bash
#SBATCH --job-name=Train_ExampleScript
#################RESSOURCES#################
#SBATCH --partition=48-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
############################################
#SBATCH --output=Eval_ExampleScript.out
#SBATCH --error=Eval_ExampleScript.err
#SBATCH -v
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ipl
###########################################
srun python eval.py --frozen_gen_ckpt ./pre_stylegan/stylegan2-ffhq-config-f.pt \
                                        --adapted_gen_ckpt ./output/disney-withDiffusersReq/checkpoint/generator/000300.pt \
                                        --source_model_type "ffhq" \
                                        --auto_compute \
                                        --output_dir ./eval/disney-withDiffusersReq \