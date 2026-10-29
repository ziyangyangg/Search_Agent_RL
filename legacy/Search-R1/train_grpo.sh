#!/bin/bash
# v2.0 verl GRPO 训练脚本（多卡真实复现，对齐项目文档04 §6.1）
# = Search-R1 原版 train_grpo.sh + 实验记录 B.20/B.23/B.24/B.27.1 的实测修改
# ⚠️ 本脚本 n_gpus_per_node=7，需要 ≥7 张卡。单卡用 train_smoke_1card.sh。
# 唯一相对 doc04 §6.1 的改动：BASE_MODEL 指向 autoDL 真实路径（doc04 写的 /root/hf_models 在 autoDL 不存在）
#* 需要修改
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl_env

# ===== 环境变量（实验记录 B.20/B.24，"缺一不可"清单）=====
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy
export no_proxy='*'

# ===== 路径配置（B.27.1）=====
export BASE_MODEL=/root/Search-R1/hf_models/Qwen2.5-3B-Instruct
export EXPERIMENT_NAME=hotpotqa-grpo-qwen2.5-3b-it-bm25-step
export TRAIN_DATA_DIR=/root/Search-R1/data/nq_hotpotqa
export TEST_DATA_DIR=/root/Search-R1/data/nq_hotpotqa

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=504 \
    data.val_batch_size=252 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=252 \
    actor_rollout_ref.actor.ppo_micro_batch_size=14 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=28 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    +trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=50 \
    trainer.project_name=DeepSearch-R1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    trainer.default_local_dir=/root/Search-R1/verl_checkpoints/$EXPERIMENT_NAME
