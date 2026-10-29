# 源码审计与证据边界

本文保留 2026-10-22 的历史审计；10 月 29 日新增缺陷与补丁范围见[已知问题](known-issues.md)。

审计日期：2026-10-22。依据外接硬盘的实际 Python、Shell、配置、数据与文件元信息。
未采用水印 PDF、商业宣传、购买指南或其中的实验叙述作为结论依据。

## 项目实际是什么

主体是 Search-R1 的本地改编：Qwen2.5-3B-Instruct 在最终答案 EM 奖励下通过 GRPO
学习交替推理、搜索、读取资料和回答。训练服务采用 Ray/FSDP/vLLM，检索侧为 Lucene
BM25；独立评估侧使用小语料和 bm25s。`eval/src/train_grpo_agent.py` 是另一条 LoRA
训练实现，其损失没有传入 old logprob，实际走带组内基线的策略梯度，不能称为与正式主线等价。

直接核对到独立评估集 500 行、小语料 41,897 行；15 份 `main_ppo.log` 均为 0 字节。
`global_step_100` 存在完整模型结构与权重分片，但这些文件不证明任何具体能力分数或训练过程。
原始项目未发现可用的 Git 历史。新项目不采用未经重跑的历史成绩作为 benchmark 结果。

## 关键发现

| 优先级 | 源码位置 | 可复现问题 | 本次处理 |
|---|---|---|---|
| P0 | `generation.py:run_llm_loop` | 构造 active batch 和 GPU padding batch 时丢失 `meta_info`；验证配置的 `do_sample=False` 可失效 | 已生成保留 metadata 的补丁；GPU 待测 |
| P0 | `main_ppo.py:RewardManager` | 从 prompt＋工具观察＋回答整段抽答案，跨越文本来源边界 | 补丁按 `info_mask` 仅选择 assistant-owned token |
| P0 | `qa_em.py:extract_solution` | 单个完整答案返回 None，隐含依赖 prompt 中的示例答案 | 补丁接受单个答案并隔离 observation |
| P1 | `generation.py:_postprocess_responses` | 只要出现 search 就优先截 search，即使 answer 更早出现 | 补丁选择最早完整动作 |
| P1 | `evaluate.py:score` | 原答案与截短答案分别和 gold 比较后取最大，不能等同固定预测的标准 EM | 新评估使用单一固定预测；旧指标不冒名替换 |
| P1 | `ray_trainer.py`／dataset | `extra_info.index` 跨来源可能碰撞，错误合并 GRPO 组 | 补丁在 repeat 前赋本 batch 独立 group ID |
| P1 | 验证 DataLoader | `drop_last=True` 遗漏尾部样本 | 新评估逐题全部计数；旧训练器尚需独立修改和 GPU 验证 |
| P1 | checkpoint／trainer | 无 optimizer/scheduler/RNG/sampler 的恢复；`resume_mode=auto` 未读取 | 明确称为权重重新启动，不伪装完整恢复 |
| P2 | 检索客户端 | 无 timeout、HTTP 状态检查；空查询也请求服务 | 已生成 timeout／错误／空批次补丁 |
| P2 | 轻量检索器 | 任意异常可静默换成另一种检索算法 | 新检索后端明确命名、失败显式暴露 |
| P2 | 打包与运行脚本 | 包名占用 verl、裸 sibling import、硬编码 root／conda／7 卡 | 新应用包独立，模型和资产路径外置 |

源码复现示例：旧 `extract_solution('<answer>London</answer>')` 返回 None；
旧 `score('London. Paris', 'London')` 的 EM 为 1。新严格 EM 为 0。
工具文本中的答案标签可被旧奖励抽取；这只是代码漏洞证明，并不证明历史训练利用过该漏洞。

## 哪些解释现在不能成立

不能将训练后的模型文件等同于能力提升证据。需要固定推理协议的训练前后配对结果。
不能将训练侧验证与独立小语料评估直接比较；语料、检索、解码、轮数、样本分母和指标均可能不同。
不能从当前文件证明“训练曾发生崩溃”，更不能证明某个 KL 系数、entropy 或算法是其原因。
不能将换强模型后的收益归因于 RL。至少区分模型规模、工具协议、训练状态三个因素。

## 本次补丁边界

补丁只针对当前快照中四个指定源码文件。它默认生成 diff 和 SHA-256 manifest，不修改原项目。
应用前须源文件与已审阅 diff 一致，并保存备份；代码漂移会拒绝应用，不做猜测替换。
Python 测试覆盖奖励来源、metadata、动作解析、检索错误、补丁预条件与回滚。
这些 CPU 检查不替代 GPU 集成验证，不证明训练效率或新模型能力。

仍需实测：长上下文下 rollout 与重算概率的条件一致性、采样截断分布、训练/推理引擎差异、
完整恢复、验证尾批、无效轨迹、loss 归一化，以及新部署模型的模板与工具调用兼容性。
