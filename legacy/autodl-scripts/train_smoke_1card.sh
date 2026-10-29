#!/bin/bash
# 单卡测试训练脚本（1×A800-80G）—— 与项目文档04 §6.1 的算法参数一致，
# 只把"多卡 + 大 batch"改成"单卡 + 小 batch + 最多 1000 步"，用来验证：
#   GRPO 学习链路在这台单卡机器上确实能跑通（advantages 有 spread、actor/pg_loss>0、reward 非零）
# GRPO、n_agent、KL、学习率和 state masking 等核心算法参数与正式脚本保持一致；
# batch、GPU 数、显存参数、训练数据量、步数、保存频率和验证设置按单卡用途调整。
# ⚠️ 本脚本不追求 44.80%（小 batch 方差大），只验证"训练正确运行"。要复现 44.80% 用 train_grpo.sh（需 ≥7 卡）。
# 16000 题 / batch 16 = 1000 个 batch；一轮数据自然结束，不额外触发结束验证。
set -e
cd /root/Search-R1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl_env

# ===== 环境变量（与正本一致）=====
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # 若 vllm 初始化报 flash_attn 相关错，改成 XFORMERS（本环境实测 XFORMERS 亦可）
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy
export no_proxy='*'

export BASE_MODEL=/root/Search-R1/hf_models/Qwen2.5-3B-Instruct
export EXPERIMENT_NAME=smoke-1card
export TRAIN_DATA_DIR=/root/Search-R1/data/nq_hotpotqa
export TEST_DATA_DIR=/root/Search-R1/data/nq_hotpotqa

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=16000 \
    data.val_data_num=null \
    data.train_batch_size=16 \
    data.val_batch_size=5 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
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
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=smoke-test \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1002 \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    trainer.default_local_dir=/root/Search-R1/verl_checkpoints/$EXPERIMENT_NAME \
    2>&1 | tee /tmp/train_smoke.log

# ===== 验收标准（单卡冒烟通过 = 下面都满足）=====
# 1. log 出现 "step:1" 且带 metrics
# 2. critic/advantages/max 明显 > 0（如 >0.5，因 n_agent=5 → GRPO group 有 spread）—— 这是与上一版漏 n_agent 的关键区别
# 3. actor/pg_loss != 0
# 4. env/number_of_valid_search > 0（BM25 检索链路通）
# 5. critic/score/mean 可能为 0（batch=16 小，base model 未必答对）——不影响"链路通"判定
