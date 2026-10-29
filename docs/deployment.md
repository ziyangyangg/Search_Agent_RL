# 部署与办公任务边界

当前版本提供可测试的检索与文件工作区 Agent 核心。它能连接支持工具调用的 Chat Completions 兼容模型服务，读取受限文本文件、搜索已建立索引的语料，并在显式授权后创建新文件。它尚未实现邮件、日历、浏览器、桌面 GUI、MCP 客户端或企业账号连接器。

真实模型端点尚未在本次整理中验证。离线 demo 与单元测试用于验证软件行为，不代表模型能力、办公任务成功率或部署效果。

## 1. 最小运行方式

以下命令均在项目根目录执行，需要 Python 3.10 及以上。工作区目录必须已经存在。

```bash
PYTHONPATH=src python3 -m agent_workbench.cli demo \
  --workspace examples/office \
  --trace runs/demo-001.json
```

此命令使用固定响应，验证工具调用与轨迹保存。轨迹文件采用独占创建：重复运行时使用新的文件名。

连接真实服务时，将下面的模型名替换为服务实际提供的 model ID。示例地址指向本机服务，不会启动模型服务器。

```bash
PYTHONPATH=src python3 -m agent_workbench.cli run \
  '读取 meeting-notes.md，概括待办事项并注明文件来源。' \
  --workspace examples/office \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL_ID \
  --trace runs/read-notes-001.json
```

访问需要认证的服务前，通过环境变量设置密钥；默认读取 `AGENT_API_KEY`，也可用 `--api-key-env` 指定变量名。不要把密钥写进配置、命令行参数或仓库。远程端点必须使用 HTTPS；HTTP 仅允许回环地址。

## 2. 当前工具与写入授权

| 工具 | 参数 | 行为 |
| --- | --- | --- |
| `workspace_list_files` | 可选 `path` | 列出直接子项，最多 500 项；不递归扫描 |
| `workspace_read_text` | `path`，可选 `max_chars` | 读取 UTF-8 文本，默认 20,000 字符，上限 100,000 字符 |
| `workspace_write_text` | `path`、`content` | 创建新文件；不覆盖已有文件、不自动创建父目录 |
| `corpus_search` | `query`，可选 `topk` | 仅在传入 `--corpus-index` 后注册；检索本地 SQLite FTS5 索引 |

工作区文件目前只支持 `.md`、`.txt`、`.csv`、`.json`。隐藏路径、常见凭据文件、路径穿越和符号链接均被拒绝。工具参数按注册的 JSON Schema 子集校验；未知参数及无效类型不会传给处理函数。

默认禁止文件写入。只有明确要在本次运行中授权创建工作区文件时，才添加：

```bash
PYTHONPATH=src python3 -m agent_workbench.cli run \
  '读取 meeting-notes.md，将待办事项整理到新文件 action-items.md，并注明来源。' \
  --workspace examples/office \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL_ID \
  --allow-workspace-writes \
  --trace runs/action-items-001.json
```

该开关授权本次运行中的所有已注册工作区写工具。当前没有交互审批界面，也没有“暂停后继续执行”的审批恢复机制。未授权写入返回 `approval_required`；运行器记录待审批状态，不执行写入。再次运行前应审阅任务与工作区，并选择新的轨迹文件。

路径检查与 `O_NOFOLLOW`、独占创建是应用层防护，不能替代操作系统沙箱；并发恶意修改目录可能绕过部分检查。面向不可信用户或加入代码执行工具时，需要独立进程或容器、最小挂载、网络限制和独立凭据。Docker 隔离也需正确配置，不能仅凭使用容器宣称生产安全。

## 3. 语料检索

语料输入是 JSONL，每行至少包含唯一 `id` 和字符串 `contents`：

```json
{"id":"policy-001","contents":"这里是文档正文。"}
```

```bash
PYTHONPATH=src python3 -m agent_workbench.cli index \
  --corpus /path/to/corpus.jsonl \
  --out runs/corpus-001.sqlite

PYTHONPATH=src python3 -m agent_workbench.cli run \
  '搜索差旅报销规定，回答时引用文档 ID。' \
  --workspace examples/office \
  --corpus-index runs/corpus-001.sqlite \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL_ID \
  --trace runs/policy-search-001.json
```

SQLite FTS5 面向小语料，不等同于旧项目的 Lucene BM25、BM25s 或稠密检索后端；分词、排序和评估结果均不能直接混用。当前查询采用词项 OR 检索，单段返回最多 6,000 字符，截断会显式标记。

## 4. 接入更强模型

模型名称和端点均由参数提供，因此运行器不绑定 Qwen2.5-3B。可接入合适的本地 vLLM 服务或云端兼容服务，但兼容性必须逐模型、逐服务版本核对：

1. 接受 `messages`、函数工具定义、`tool_choice=auto`、`parallel_tool_calls=false`、`temperature` 和 `max_tokens`。
2. 工具调用返回非空、唯一的 `id`，正确的函数名，以及内容为 JSON 对象的 `function.arguments` 字符串。
3. 下一轮能接受 `role=tool` 与匹配的 `tool_call_id`，并据真实结果继续执行或回答。
4. chat template、工具解析器及模型原生工具协议匹配；不要复用旧 Search-R1 的特殊标签停止条件。
5. 当前处理文本内容、标准函数工具调用，以及可选的 `reasoning_content` 字段；该字段会在工具结果回传时保留。`enable_thinking` 和 `reasoning_effort` 参数仍须由具体服务支持。多模态输入、Responses API、流式输出和其它服务专属字段需要单独适配。

vLLM 官方提供兼容服务接口；Qwen-Agent 官方实现也将模型、工具与 Agent 分开，并支持云服务及兼容端点。它们证明这种接口分离可行，不证明任意 model ID 都能在当前客户端直接工作：

- [vLLM 服务接口文档](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
- [Qwen-Agent 官方仓库](https://github.com/QwenLM/Qwen-Agent)

先做单工具、两轮工具结果回传及创建文件测试，再运行能力评估。升级模型时固定工具实现、任务集、提示词和预算；同时记录任务成功率、延迟与 token 使用量，避免把增加预算造成的变化归给模型本身。

## 5. 推理接口与强化学习接口

兼容 API 解决的是推理接入。GRPO 等参数训练还需要可训练权重、token ID、采样时的 log probability、policy version、生成/工具观察来源掩码，以及优化器和训练状态。

当前文本消息轨迹不能自动替代这些训练张量，托管模型也不能仅靠更换端点便加入本地 GRPO 更新。强模型可在服务条款允许的范围内提供演示轨迹或教师数据；采集数据、监督微调与在线 RL 应分别说明。旧 Search-R1 训练线与新运行器的训练适配仍需独立实现和验证。

## 6. 何时可以称为办公 Agent

当前 `completed` 表示模型结束工具循环并返回了文本，不表示办公任务已经被独立验证。生成报告后应检查文件确实存在、内容符合任务、来源可追溯；涉及业务系统时应验证数据库或应用最终状态，而不是只判断回复“看起来完成了”。

推荐按以下顺序扩展，并为每一步保留独立评估：

| 阶段 | 待补能力 | 验收依据 |
| --- | --- | --- |
| 文件研究助理 | 文档格式解析、产物校验、来源检查 | 文件内容与约束验证；受攻击时的任务效用 |
| 应用 API 助理 | 连接器、账号范围、幂等写入、审批与恢复 | 应用最终状态、重复执行、权限边界 |
| 浏览器或桌面助理 | 浏览器/GUI 观察与动作适配 | 对应环境任务验证器、长任务可靠性 |

以下公开环境可为后续接口与评估提供依据，目前均未集成：

- [AppWorld](https://github.com/StonyBrookNLP/appworld)：交互应用和可执行任务环境，适合验证应用 API 工作流。
- [WorkArena](https://github.com/ServiceNow/WorkArena)：ServiceNow 知识工作与组合任务；需要获得相应实例访问权限。
- [AgentDojo](https://github.com/ethz-spylab/agentdojo)：同时评估正常任务效用和工具返回内容中的提示注入攻击。
- [τ-bench 评估说明](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)：说明如何按应用最终状态及所需沟通内容评价任务，避免把单一参考工具序列误当唯一正确路径。

## 7. 数据流与运行记录

使用远程模型时，用户任务、已读取文件、检索段落和完整历史消息会发送到指定模型服务。API 密钥不写入轨迹，但轨迹会包含任务、工具参数、文件内容与模型输出；它可能比最终回答包含更多敏感信息。当前没有自动脱敏、加密存储、多租户身份鉴权或保留期限管理。

因此应使用经允许的数据、明确指定的工作区和可信模型端点；轨迹不要提交到公开仓库。工具权限只能约束本地动作，不能阻止已读取内容通过模型请求发往远程服务。公开实验宜使用合成或脱敏语料，记录模型、数据版本、预算和验证器，并将软件测试结果与真实能力实验分开报告。
