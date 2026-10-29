# 当前缺点与后续验收

本文区分新 Workbench 与旧 Search-R1。独立重构绕开了部分旧路径，四文件补丁也只修了明确
范围，不能据此声称旧工程全部修好。最终实测检查以[验证记录](verification.md)为准。

## 新工程仍缺少的能力

| 优先级 | 缺口 | 当前影响 | 完成条件 |
| --- | --- | --- | --- |
| P0 | 真实端点兼容性待验收 | fixture 能验证软件分支，不能证明指定 Jev/Qwen 模型、模板和服务能完成任务。 | 固定模型版本，跑通单工具、多轮回传、服务失败；保存脱敏证据。 |
| P0 | 没有业务终态 verifier | `completed` 不等于正确，仍可能答错或误称完成。 | 检查答案、引用与文件/应用最终状态，不能只看回复。 |
| P0 | 新运行器没有 RL 训练链路 | 轨迹缺真实采样 token IDs、logprobs、policy/template version。 | 接训练适配器，验证 observation mask 与分组，再做 GPU smoke test。 |
| P1 | Jev 依赖预先准备的候选 | 不能开放生成参数或计划；候选缺失时只能升级。 | 与看见相同候选的单模型基线对照，再评估自动候选和计划复用。 |
| P1 | 没有实测的协作收益 | fixture 少一次生成调用不能证明端到端加速或准确率提升。 | 报成功率、快速路径覆盖/条件错误率、p50/p95 延迟、token 与费用。 |
| P1 | 上下文不是 token 级管理 | 字符上限只约束序列化请求规模；不能保证模型窗口不超限，没有自动压缩。 | 固定 tokenizer/template，计算输入与输出 token 预算，测试长轨迹。 |
| P1 | 中断后不能完整恢复 | QA 逐题 flush，但没有自动续评；任务状态和工具幂等键不持久化。 | 按完成 ID 续评，定义写入重放策略，避免重复副作用。 |
| P1 | 路径检查不是 OS 沙箱 | 应用约束不能完全隔离并发恶意文件系统变化。 | 不可信工具用隔离进程/容器和独立权限测试。 |
| P1 | 运行轨迹包含资料 | 任务、工具返回、reasoning 可能含私有文档；启用 Jev 会向第二服务发送部分上下文。 | 明确数据流、保留期和访问权限；公开时仅用合成或授权数据。 |
| P2 | 小语料检索边界 | SQLite FTS5 的词项查询不能证明中文、跨语言、语义或大规模检索质量。 | 在目标语料上评估召回；替换检索后端独立记录。 |
| P2 | 办公工具有限 | 仅文本列表、读取、创建和语料检索，未集成邮件/日历/浏览器/Office 二进制。 | 逐任务接入工具、授权和 verifier，不用问答分数替代办公验收。 |
| P2 | QA 指标覆盖有限 | 短答案 EM/F1 不能覆盖长程任务，三模式预算也不同。 | 成本与准确率共同报告，应用任务另设终态成功率与约束违规指标。 |

## 旧源码已确认而未全部覆盖的问题

位置来自 2026-10-29 外接盘只读审计；行号对应当时快照。这些文件不是新应用的推理依赖。

| 旧源码位置 | 确认的问题 | 本仓库覆盖范围 |
| --- | --- | --- |
| `eval/src/evaluate.py:181` | 对原答案/截短答案分别看 gold 后取高分，不能当作固定预测标准 EM/F1。 | 新评估固定预测；补丁不修改旧评估脚本。 |
| `eval/src/evaluate.py:175`、`eval/src/rewards.py` | 全轨迹抽答案可能把检索段落标签当成模型答案。 | 新消息来源分离；正式奖励补丁用 token provenance，未覆盖全部独立评估/LoRA 路径。 |
| `infer.py` | 无回合/上下文上限，累计历史解析可能重放旧搜索，固定 Qwen2.5 EOS。 | 新运行器有显式预算和结构化协议；旧文件未改。 |
| `search_r1/search/retrieval_server.py:341` | 关闭 scores 后仍二元解包，默认请求会报错或错配。 | 不在四文件补丁中；需 query 批次与开关矩阵测试。 |
| `search_r1/search/rerank_server.py:43` | 每个 query 遍历所有 query 的文档，跨查询污染并增加评分量。 | 未修；应只访问对应候选集合。 |
| `search_r1/search/retrieval_rerank_server.py:75` | 返回字符串 document，客户端要求 `document.contents`。 | 未修；需统一服务响应契约。 |
| `search_r1/search/google_search_server.py:153` | 默认完整内容模式使用未初始化的 `search_results`。 | 未修；需默认配置 mock 集成测试。 |
| `search_r1/search/retrieval_server.py` | FAISS `-1` 未过滤，dense encoder 无条件 CUDA。 | 未修；需小语料大 topk 与设备配置测试。 |
| 检索服务与客户端 | 多处无 timeout/状态校验，服务监听全部接口且无认证。 | 补丁仅修训练生成器的一个请求路径，未修所有服务。 |
| AutoDL 脚本 | 固定 root/conda/GPU 数；部分管道缺 pipefail，停止/清理范围较宽。 | 新推理不依赖；未整体迁移修复。 |
| 独立 LoRA trainer | 组内基线策略梯度并不等同正式 clipped GRPO；权重保存不等于完整恢复。 | 保留准确说明；新包未接该 trainer。 |
| 旧验证/恢复 | 验证丢尾批、resume 配置与实际能力不一致等问题。 | 新 QA 全量计数不代表旧训练验证已修；GPU 待测。 |

## 四文件补丁究竟修了什么

[patch_legacy.py](../scripts/patch_legacy.py) 仅处理：

- `search_r1/llm_agent/generation.py`：保留 metadata、按最早完整动作停止、检索请求检查。
- `verl/trainer/main_ppo.py`：只用 assistant-owned token 提取答案奖励。
- `verl/utils/reward_score/qa_em.py`：接受单个答案，并隔离工具 observation。
- `verl/trainer/ppo/ray_trainer.py`：repeat 前设置并保留独立 rollout group ID。

默认仅生成 diff 和审阅 manifest；应用前需匹配源码与审阅哈希并备份。CPU 回归不等于补丁
已写回原盘，也不等于 GPU 训练通过。

## 建议的验收顺序

1. 固定真实端点，完成只读任务、工具失败、多轮回传和预算退出检查。
2. 固定模型/数据/语料/检索/模板/解码/评分，分别建立旧标签与新原生工具协议基线。
3. 为一个真实文档任务建立终态 verifier，报告成功、失败与非目标修改。
4. Jev 与同候选信息的单模型比较质量、覆盖率和成本，再决定是否扩展自动规划或路由训练。
5. 接入 token 级采样记录和训练后端，GPU smoke test 通过后再比较算法。
