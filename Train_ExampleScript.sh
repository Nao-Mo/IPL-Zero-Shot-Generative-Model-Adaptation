#! /bin/bash
#SBATCH --job-name=Train_ExampleScript
#################RESSOURCES#################
#SBATCH --partition=48-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
############################################
#SBATCH --output=Train_ExampleScript.out
#SBATCH --error=Train_ExampleScript.err
#SBATCH -v
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ipl
###########################################
srun python train.py  --frozen_gen_ckpt ./pre_stylegan/stylegan2-ffhq-config-f.pt \
                                        --source_model_type "ffhq" \
                                        --output_interval 300 \
                                        --save_interval 300 \
                                        --auto_compute \
                                        --source_class "photo" \
                                        --target_class "disney" \
                                        --run_stage1 \
                                        --batch_mapper 32 \
                                        --lr_mapper 0.05 \
                                        --iter_mapper 300 \
                                        --ctx_init "a photo of a" \
                                        --n_ctx 4 \
                                        --lambda_l 1 \
                                        --run_stage2 \
                                        --batch 2 \
                                        --lr 0.002 \
                                        --iter 300 \
                                        --output_dir ./output/disney-withDiffusersReq
