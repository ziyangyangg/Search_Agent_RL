# Search Agent RL · Search Agent Workbench

一个便于阅读、测试和持续改进的**检索与文档工作区 Agent**。项目源自对 Search-R1 的复现审计，将模型接口、工具执行、评估和训练记录分开组织。

**版本 0.2.1。** 本仓库包含独立重构应用、只读候选动作的 Jev/Qwen 协作接口，以及旧 Search-R1 的审阅补丁。原始训练源码、数据、权重独立管理；当前运行器尚未接入新的 RL trainer。

## 五分钟开始

需要 Python 3.10+。macOS / Linux：

```bash
git clone https://github.com/ziyangyangg/Search_Agent_RL.git
cd Search_Agent_RL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
agent-workbench demo --workspace examples/office
python -m unittest discover -s tests -v
```

Windows 使用 `py -m venv .venv`、`.venv\Scripts\Activate.ps1` 激活，随后执行相同的 `python` 命令。离线 demo 使用固定响应，不下载模型、不调用付费 API；成功时返回 `status: completed`。运行时仅依赖标准库，安装工具另需 setuptools。

不安装应用时，在 macOS / Linux 从源码运行：

```bash
PYTHONPATH=src python3 -m agent_workbench.cli demo --workspace examples/office
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 阅读入口

| 目标 | 文档 |
| --- | --- |
| 跑通项目、接入模型与排错 | [使用指南](docs/getting-started.md) |
| 理解模块、执行流程与数据契约 | [架构和阅读顺序](docs/architecture.md) |
| 回顾此前整理与这次改进 | [项目回顾](docs/project-review-2026-10-29.md)、[更新记录](CHANGELOG.md) |
| 了解现有缺点与验收目标 | [已知问题](docs/known-issues.md) |
| 审查旧 Search-R1 | [源码审计](docs/audit.md) |
| 使用 Jev 与 Qwen 协作 | [双系统设计](docs/dual-system.md) |
| 做评估、继续 RL 研究 | [评估协议](docs/evaluation.md)、[RL 路线](docs/rl-roadmap.md) |
| 了解文件工作区和部署 | [部署说明](docs/deployment.md) |
| 继续贡献代码 | [贡献指南](CONTRIBUTING.md)、[验证记录](docs/verification.md)、[发布说明](docs/publishing.md) |

## 代码组织

以下模块均位于 `src/agent_workbench/`：

| 模块 | 职责 |
| --- | --- |
| `types.py` | 模型响应、工具调用、轨迹契约 |
| `models.py`、`jev.py` | 模型 HTTP 传输与返回值校验 |
| `cooperation.py` | 从已准备的只读候选中选动作，必要时升级推理 |
| `runtime.py` | 唯一执行循环、预算与终止状态 |
| `tools.py`、`retrieval.py` | 参数校验、工作区文本工具、SQLite FTS5 检索 |
| `evaluation.py`、`benchmark.py` | 严格评分、完整分母、逐题证据 |
| `training.py` | 带角色来源的文本轨迹导出，尚非 token 对齐训练数据 |

`scripts/` 管理安装和旧项目补丁，`tests/` 存放回归，`examples/` 提供公开合成样例，`configs/` 保存候选动作配置。模型只提议动作，执行器统一校验并执行。

## 连接真实模型

先准备支持原生函数工具调用的 Chat Completions 兼容端点，再运行：

```bash
agent-workbench run \
  '读取 meeting-notes.md，列出行动项并注明来源。' \
  --workspace examples/office \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL_ID \
  --trace runs/read-notes-001.json
```

需要认证时用环境变量 `AGENT_API_KEY`。远程服务要求 HTTPS，本机允许回环 HTTP。此命令不会启动模型服务器；模型名、模板和工具协议必须由指定端点支持。

默认只读。增加 `--allow-workspace-writes` 才能创建新文本文件；不能覆盖已有文件。每次实验使用新的轨迹文件名。

## 检索与评估

先运行无模型的合成样例：

```bash
agent-workbench index --corpus examples/corpus.jsonl --out runs/example-corpus.sqlite
agent-workbench score --predictions examples/predictions.jsonl --out runs/example-score.json
```

索引应有 3 篇合成文档；评分样例 `n=2`、strict EM=0.5，只验证评分规则。真实端点的 direct/RAG/agent 对照：

```bash
agent-workbench evaluate \
  --dataset examples/questions.jsonl --corpus-index runs/example-corpus.sqlite \
  --base-url http://127.0.0.1:8000/v1 --model YOUR_SERVED_MODEL_ID \
  --out-dir runs/qa-example-001
```

输出数据/索引哈希、逐题预测、轨迹、状态、token 数和严格指标。各模式计算预算不同，比较时同时报告成本。轨迹可能包含私有文件，不应提交到公开仓库。

## 当前边界

Jev 快速路径依赖已准备的只读候选；自动规划、学习路由和真实加速收益尚未验证。`completed` 只表示得到完整响应，任务正确性还需独立 verifier。步骤、工具数和输入字符有预算，字符上限不等同模型 token 窗口。

CPU 测试验证软件行为；真实模型/GPU训练/实际办公成功率需另行验收。旧标签协议 checkpoint 不能直接当作原生工具调用模型。历史成绩不作为本次已复现结果，详见[已知问题](docs/known-issues.md)。

## 来源与许可

遵循 Apache-2.0，保留 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。原项目：[Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[veRL](https://github.com/volcengine/verl)。模型和数据另有使用条件。仓库不包含权重、原始数据、商业指南或私有运行轨迹。
