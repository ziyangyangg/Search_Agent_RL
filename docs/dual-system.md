# Jev 与 Qwen3.8 协作：从单模型搜索走向自适应计算

项目目标更新：使用 Jev 的快速结构化判断与 Qwen3.8 的生成、规划和复杂推理协作，
在实际工具任务上改善质量—延迟—成本曲线。这里 Jev 对应 System 1；Qwen3.8 的思考模式
承担 System 2。模型名字不是认知模式的保证，Qwen 的快答模式也可作为第三种计算档位。

## 职责与接口

| 组件 | 负责 | 不能假定 |
|---|---|---|
| Jev | 固定候选中选动作、判断段落是否支持结论、发现冲突／异常、决定是否升级推理 | 不自由生成新查询、文件正文、代码或开放参数 |
| Qwen3.8 | 制定和修订计划、生成工具参数、综合证据、撰写产物、复杂纠错 | 不要求每次简单判断均重新长推理 |
| 执行器 | schema、权限、调用、错误、预算和轨迹 | 不由模型高置信度替代授权或任务验证 |
| verifier | 检查答案、文件、应用状态等实际结果 | 不以模型说“完成”为成功 |

Jev 官方文档说明其接口是 `POST https://api.typesafe.ai/v1/systemone`，传入 state 与
typed questions；与生成式 Chat Completions 接口分开。Choice 选择候选，Noul 返回是非概率。
见 [API](https://docs.typesafe.ai/api/system-one) 和
[闭集函数调用示例](https://docs.typesafe.ai/cookbooks/function_calling)。

## 理想闭环与本次已实现范围

理想闭环为：Qwen 按需生成可执行计划／候选 → 工具交互 → Jev 判断当前进展和候选动作 →
简单明确步骤直接执行，缺信息、冲突、无合适动作时交回 Qwen → 更新计划，直到可验证终态。
计划需复用；若每轮仍完整调用 Qwen，再追加 Jev，通常只会增加延迟。

0.2 已实现可测试子集：读取 JSON 中准备好的完整只读动作；Jev 从动作与 `reason` 升级选项中
选择。概率不足、API 异常、上下文过大或候选耗尽都调用 Qwen。运行器仍校验并执行动作，
快速路径不能写文件、不能重复执行同一已尝试候选。没有候选时不调用 Jev。

**尚未实现**：Qwen 自动生成可复用条件计划、学习式路由、段落核查、Qwen 快答／深思级联、
双系统 RL 训练，以及合作 Agent 的真实质量和速度评测。现有候选文件是可审阅工作流配置，
不能描述为模型已经学会自主规划。

```bash
PYTHONPATH=src python3 -m agent_workbench.cli run \
  '读取 meeting-notes.md，概括行动项并注明来源。' \
  --workspace examples/office \
  --base-url YOUR_QWEN_COMPATIBLE_BASE_URL \
  --model YOUR_QWEN38_MODEL_ID \
  --thinking on --reasoning-effort medium --max-tokens 8192 \
  --jev-model YOUR_PINNED_JEV_MODEL_ID \
  --candidates-file configs/read-notes-candidates.json \
  --trace runs/cooperative-001.json
```

运行前设置 `AGENT_API_KEY`（Qwen 端点）和 `JEV_API_KEY`。两个服务的密钥、模型 ID 和计费
分别管理；正式实验固定可用模型版本，不使用会静默更新的 alias。Qwen 请求参数需由对应
服务支持，思考模式产生的 `reasoning_content` 在工具回传时保留；见
[Qwen 模型与工具说明](https://docs.qwencloud.com/developer-guides/getting-started/latest-model)。

## 支持这个方向的证据

[TypeSafe SDE cascade](https://docs.typesafe.ai/cookbooks/sde_cascade) 给出“小模型提取→
Jev 按字段判断→必要时升级 reasoning model”的具体实现及其小规模成本—质量演示。
[RAG passage cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) 将 Jev 放在
检索与生成之间，判断相关性、可用证据、冲突与注入。这些是厂商提供的可实现模式，
不构成 Jev＋Qwen3.8 在我们任务上已优于 Qwen 单模型的证据。

Jev 的 `confidence` 是概率分布的集中程度统计量，不是此动作“正确的概率”。
`P(选某动作)` 也不自动等于 `P(整个任务成功)`。阈值需按动作类型、数据分布与失败代价验证；
当前 0.9 仅为实验初值，不提供 90% 或 99% 正确率保证。见
[官方 confidence 定义](https://docs.typesafe.ai/confidence)。

## 为什么可能更快、更准，以及如何证伪

若省掉 k 次原 Qwen 调用，增加 n 次 Jev 判断与额外计划开销，在相同工具耗时下近似有：

`节省时间 ≈ k × 单次Qwen耗时 − n × 单次Jev耗时 − 额外规划耗时`。

端到端必须实测网络 RTT、排队、prefill、长上下文和重试，不能把厂商单次决策速度当成
整个 Agent 的加速比。批量判断仅适用于共享相同状态的独立问题，不能合并有先后依赖的决策。

准确度收益的假设是：便宜核查能阻止部分错误延续，复杂情况仍交给推理器。
它也可能损害准确度：错误高置信选择、候选缺失、过早停止、误删关键证据或升级成本过高。
必须报告覆盖率、快速路径条件错误率、错误升级率和候选覆盖不足；不能只报平均成功率。

最小对照：A=Qwen 单模型；B=Qwen＋Jev 核查（不跳过 Qwen）；C=Jev 选择候选＋Qwen 按需介入；
D=可选的 Qwen 快答＋Jev 核查＋Qwen 深思。固定任务、工具、候选可见信息与最终验证器；
为避免规划先验混淆，额外比较“Qwen 也可使用同一候选集”的基线。记录任务成功率、严格 EM、
工具错误、p50/p95 延迟、Qwen 调用数、Jev 调用数、两者 token 和实际费用。

## 下一步 RL 的对象

Jev 在公开 API 下视为冻结决策服务，不假定可更新其参数。若 Qwen 使用可训练权重，
可以优化其计划、候选质量和纠错行为；路由策略则根据最终任务奖励和计算成本学习。
若使用托管 Qwen API，需另外确认训练接口，普通推理密钥不提供本地梯度更新。

路由真正应学习的是“升级 Qwen 带来的预期收益是否超过时间和费用”，而非机械地提高 Jev
置信度。奖励可包含任务可验证成功、计算成本和非目标修改代价；先建立无训练的合作基线，
再比较路由学习与 Qwen 参数训练，避免将换模型、加工具、加预算的收益归给 RL。

当前软件测试只证明接口与降级机制的行为；没有实际访问收费端点，也没有观测真实加速或准确度收益。
