# 使用指南

按“离线运行 → 索引和评分 → 单模型 → Jev 协作”的顺序开始。全部命令从仓库根目录执行。

## 1. 安装和离线检查

依 README 创建 Python 3.10+ 虚拟环境，然后执行：

```bash
python -m pip install -e .
agent-workbench --help
agent-workbench demo --workspace examples/office --trace runs/demo-001.json
python -m unittest discover -s tests -v
```

demo 通过工具列出示例目录，再返回固定文本。轨迹 JSON 的 `messages` 是实际上下文，`events` 是模型与工具事件，`status` 是退出原因。重复运行改用 `demo-002.json`，已有轨迹不会覆盖。

无网络安装条件时用 README 的源码运行方式。应用运行时无需第三方包，安装流程可能需要获取构建工具。

## 2. 索引和评分

```bash
agent-workbench index --corpus examples/corpus.jsonl --out runs/example-corpus.sqlite
agent-workbench score --predictions examples/predictions.jsonl --out runs/example-score.json
```

预期索引 3 篇文档；评分为 `n=2`、strict EM=0.5。语料每行包含唯一 `id` 与字符串 `contents`。当前后端是 SQLite FTS5；中文分词和召回需在目标语料上检验，不能与旧 Lucene/BM25s 混称。

## 3. 单模型任务

先准备支持原生工具调用的兼容端点。如需认证，通过环境变量设置 `AGENT_API_KEY`；程序不会自动加载 `.env`。下面的 model ID 须换成服务实际提供的值。

```bash
agent-workbench run \
  '读取 meeting-notes.md，概括行动项并引用文件路径。' \
  --workspace examples/office \
  --base-url http://127.0.0.1:8000/v1 --model YOUR_SERVED_MODEL_ID \
  --max-steps 8 --max-tool-calls 12 --max-context-chars 120000 \
  --trace runs/model-001.json
```

先验证读取、工具回传和最终回答，再尝试创建文件。`--allow-workspace-writes` 授权本次运行创建新文件，不覆盖文件、不自动创建父目录。启用检索加 `--corpus-index runs/example-corpus.sqlite`。

思考模式参数需由对应服务支持；不确定时保持默认。兼容协议见[部署说明](deployment.md)。退出码 0 表示 `completed`，2 表示未完成或配置错误；业务正确性需另验证。

## 4. Jev 候选动作选择

先跑通单模型。分别准备 `AGENT_API_KEY` 与 `JEV_API_KEY`，模型和服务地址分别管理。

```bash
agent-workbench run \
  '读取 meeting-notes.md，概括行动项。' \
  --workspace examples/office \
  --base-url YOUR_QWEN_COMPATIBLE_BASE_URL --model YOUR_QWEN_MODEL_ID \
  --jev-model YOUR_PINNED_JEV_MODEL_ID \
  --candidates-file configs/read-notes-candidates.json \
  --trace runs/cooperative-001.json
```

Jev 从完整只读候选中选择，候选耗尽、概率不足或服务出错时升级生成模型。默认 0.9 是实验阈值，不是正确率保证。当前没有自动生成计划。见[双系统设计](dual-system.md)。

## 5. 排错

| 现象 | 处理 |
| --- | --- |
| 找不到命令 | 激活正确虚拟环境并安装；或改用源码运行方式 |
| `FileExistsError` | 换新输出文件/评估目录，原结果不覆盖 |
| `model_error` | 检查轨迹错误类型、地址、密钥、模型与工具协议 |
| `context_budget` | 缩小任务/累积资料或审慎调整字符上限；仍需满足模型 token 窗口 |
| `step_budget` / `tool_budget` | 检查重复调用和证据不足，再调整任务或预算 |
| `needs_approval` | 任务要求写入但未授权；审阅后用新轨迹重新运行 |
| `no such module: fts5` | 换用包含 SQLite FTS5 的 Python 构建 |
| 检索无结果 | 核对 query、分词和语料，不静默改检索算法 |

## 6. 与外接盘旧项目配合

新应用独立运行，不依赖外接盘。需要引用旧资产时使用 `scripts/install_on_drive.py` 旁路安装；命令见[工程重组](reorganization.md)。该脚本不复制模型和数据。

旧补丁只适用于已审计快照。先生成并审阅 diff/manifest，源文件漂移会拒绝应用。缺少旧快照的读者可以运行新应用，但不能仅凭补丁复原整个旧训练环境。
