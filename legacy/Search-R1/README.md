# Search-R1 本地研究快照：阅读与复现入口

本目录包含原盘实际存在的 Search-R1/veRL、检索服务、独立评估、数据构建、训练脚本与配置。
原快照说明已原样移到 [README.upstream.md](README.upstream.md)。
从仓库根目录的 [README](../../README.md) 运行新应用；此处环境独立，适合研究和迁移旧训练线。

## 阅读顺序

1. `eval/src/agent.py`、`prompts.py`：理解旧标签协议和轻量执行循环。
2. `search_r1/llm_agent/generation.py`：理解正式训练中的搜索 rollout 和来源 mask。
3. `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py`：奖励、分组和训练调度。
4. `verl/trainer/config/ppo_trainer.yaml` 与 `train_smoke_1card.sh` / `train_grpo.sh`：运行配置。
5. `eval/src/evaluate.py`、`rewards.py`、`train_grpo_agent.py`：独立评估与另一路 LoRA 实验。

已知评分、检索契约、恢复和脚本问题见[审计](../../docs/audit.md)及[已知问题](../../docs/known-issues.md)。
独立 LoRA 实验采用组内基线策略梯度，不应与正式 clipped GRPO 混称。

## 为什么没有上传数据和权重

源码仓库不包含外接盘上的 parquet、Lucene 索引、HotpotQA 原始语料与约 13.6GB 权重。
这些属于独立资产，既有许可条件和资源要求，也不适合普通 Git 分发。来源入口：

- HotpotQA：<https://huggingface.co/datasets/hotpotqa/hotpot_qa>，配置为 `distractor`。
- Search-R1 官方模型/数据集合：<https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5>。
- Wiki 语料与索引：`scripts/download.py`，对应 `PeterJinGo/wiki-18-corpus` 和 `PeterJinGo/wiki-18-e5-index`。
- 基础模型：`Qwen/Qwen2.5-3B-Instruct`。原盘 `global_step_100` 为本地 checkpoint，不假定有公开下载地址。

## 重建小语料的实际入口

在单独的数据准备环境安装 `datasets`，从**本目录**执行以下脚本。它会下载数据并生成新文件：

```bash
HF_ENDPOINT=https://huggingface.co python eval/src/build_dataset.py \
  --split train --max_samples 4000 \
  --qa_out eval/data/train/hotpotqa_train.jsonl \
  --corpus_out eval/data/corpus/hotpotqa_corpus.jsonl --corpus_mode overwrite

HF_ENDPOINT=https://huggingface.co python eval/src/build_dataset.py \
  --split validation --max_samples 500 \
  --qa_out eval/data/eval/hotpotqa_dev.jsonl \
  --corpus_out eval/data/corpus/hotpotqa_corpus.jsonl --corpus_mode append
```

结果格式与新 Workbench 评估器兼容，但数据集 revision、行顺序和处理版本必须固定后才能复核历史结果。
不能只用条数相同证明恢复了原来的实验。已有本地 parquet 时，可用 `eval/src/build_dataset_local.py`。
新 Workbench 可直接索引上述 corpus，并对上述 dev 数据做严格评估；命令见根目录指南。

## 训练与旧脚本的边界

正式训练依赖 Linux/CUDA、兼容的 torch/vLLM/veRL、模型、训练 parquet 和独立检索服务。
根目录应用环境不安装这些依赖。请逐项检查 `requirements.txt`、`pyproject.toml`、训练 YAML 和启动脚本。
`train_grpo.sh` 原设定使用多卡，`train_smoke_1card.sh` 只是单卡链路检查，不能保证任意 GPU 都可运行。
AutoDL 脚本有 `/root/Search-R1`、conda 路径和设备假设，使用前需配置；清理/停止脚本也应先审阅。

历史 README 中的 `retrieval_launch.sh`、根目录 `train_ppo.sh` 与部分数据脚本并不存在于原快照，
不可直接照抄。实际保留的入口包括 `retrieval_launch_bm25.sh`、`train_smoke_1card.sh`、`train_grpo.sh`，
但它们仍受审计中记录的路径、依赖和逻辑问题影响。本次未执行 GPU 训练。

## 先审阅补丁，再在副本应用

从仓库根目录生成四文件 diff：

```bash
python scripts/patch_legacy.py \
  --source-root legacy/Search-R1 \
  --output /tmp/search-r1-reviewed.diff
```

默认不修改源码。需要应用时先审阅 diff，再使用脚本帮助中的 `--apply` 和新的备份目录。
该补丁只修四个文件，不包括所有检索服务、独立评估和 LoRA trainer；应用后仍需 GPU 集成验证。
保留原文件的许可见 [LICENSE](LICENSE) 与 [Notice.txt](Notice.txt)。
