#!/bin/bash
#SBATCH --job-name=Stack_Cube_42_newlayer_newsetting
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s327313@studenti.polito.it
#SBATCH --output=stack_Cube_42_newlayer_newsetting.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a40
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100GB


source ~/miniconda3/etc/profile.d/conda.sh
conda activate xirl_py39

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


PORT=$((29000 + SLURM_JOB_ID % 1000))

# Remap CUDA_VISIBLE_DEVICES to match SLURM allocation
# export CUDA_VISIBLE_DEVICES=1,2
# echo "Remapped CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# USEFULL COMMAND FOR EXECUTION

# Execution code for INEST-IRL EGOCENTRIC
# Algo accepted values : 'distance_to_goal', 'goal_classifier', 'inest', 'inest_knn', 'state_intrinsic', 'reds'
# They must be changed also in rl.py if modified here.
# python rl_xmagical_learned_reward.py \
#   --pretrained_path /leonardo/home/userexternal/lianniel/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_EGO_6Subtask\
#   --seed 12 \
#   --wandb \
#   --algo inest

# Execution code for INEST-IRL ALLOCENTRIC
# python rl_xmagical_learned_reward.py \
#   --pretrained_path /leonardo/home/userexternal/lianniel/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_ALLO_6Subtasks\
#   --seed 12 \
#   --wandb \
#   --algo inest

python test.py

# Execution code for ENVIRONMENTAL REWARD 
python rl_xmagical_env_reward.py --embodiment gripper --seeds 1 --wandb

# ENV_ID="StackPyramid-v1"

# python -m mani_skill.utils.download_demo ${ENV_ID}

# Pretraining command
# python pretrain_xmagical_same_embodiment.py --embodiment gripper --algo xirl --unique_name --wandb

# MULTIGPU TRAINING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/Egocentric/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_EGO_6Subtask --seeds 1 --wandb --name_test Egocentric_6SubtaskXirl_Curriculum_Normal

# # ALLOCENTRIC XIRL -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_ALLO_6Subtasks --seeds 1 --wandb --name_test Allocentric_Xirl_Curriculum_20MTraining

# # ALLOCENTRIC DINOV2 -> RUNNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=dinov2_embodiment=gripper --seeds 1 --wandb --name_test Allocentric_Dinov2_6Subtasks_20MTraining

# # ALLOCENTRIC REDS -> RUMNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=reds_embodiment=gripper --seeds 1 --wandb --name_test Allocentric_Reds_6Subtasks_20MTraining

# # ALLOCENTRIC HOLDR -> RUNNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper --seeds 1 --wandb --name_test Allocentric_HolDR_6Subtasks_20MTraining

# # ALLOCENTRIC HOLDR_CONTRASTIVE -> RUNNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper_Contrastive --seeds 1 --wandb --name_test Allocentric_HolDR_Contrastive_6Subtasks_20MTraining

# # EGOCENTRIC XIRL -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_EGO_6Subtask --seeds 1 --wandb --name_test Egocentric_Xirl_6Subtask20MTraining 

# # EGOCENTRIC DINOV2 -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=dinov2_embodiment=gripper_EGO --seeds 1 --wandb --name_test Egocentric_Dinov2_6Subtask_20MTraining

# # EGOCENTRIC REDS -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=reds_embodiment=gripper_EGO --seeds 1 --wandb --name_test Egocentric_Reds_6Subtask_20MTraining

# # EGOCENTRIC HOLDR
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper_EGO --seeds 1 --wandb --name_test Egocentric_HolDR_6Subtask_20MTraining

# # EGOCENTRIC HOLDR CONTRASTIVE
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper_Contrastive_EGO --seeds 1 --wandb --name_test Egocentric_HolDR_Contrastive_6Subtask_20MTraining_Real

#VECTENV TRAINING
# torchrun --nproc_per_node=2 rl_xmagical_learned_reward_multi_vectEnv.py --pretrained_path /home/liannello/xirl/experiment_results/Egocentric/pretraining/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper=EGO_SubtaskXirl --seeds 1 --wandb --name_test 0.999G-MultiEnv-EGOEGO-5   

#ENV REWARD


# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_env_reward_multi_vectEnv.py --embodiment gripper --seeds 1 --name_test 20MillionMultiGPUENV
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT  rl_xmagical_env_reward_multi.py --embodiment gripper --seeds 1 --name_test Allo-EnvReward-20Million-Video
# python generate_plot.py
# python test_embedding.py --experiment_path /home/liannello/xirl/experiment_results/Egocentric/pretraining/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper=EGO_SubtaskXirl

# python test_embedding_reds.py --experiment_path /home/liannello/xirl/experiment_results/Allocentric/pretraining/dataset=xmagical_mode=same_algo=reds_embodiment=gripper=ALLO_Reds

# torchrun --nproc_per_node=2 test_DDP.py
