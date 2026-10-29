# 验证记录

版本：0.2.1。日期：2026-10-29。这里记录软件验证，不将其当作模型能力实验。

## 发布前检查

| 检查 | 结果与范围 |
| --- | --- |
| 标准库 unittest | 72 项通过；包括原有 52 项、安装白名单 4 项、执行预算 4 项、适配器边界 12 项新增回归 |
| Python 语法 | src/scripts/tests 全部 Python 源文件可解析 |
| 文档链接 | 根目录和 docs 内 Markdown 的相对链接检查通过 |
| wheel 构建 | 0.2.1 构建成功 |
| 独立安装 | 临时虚拟环境中仅从本地 wheel 安装成功 |
| 安装后入口 | 离开源码目录并移除 PYTHONPATH 后，help/demo/index/score 通过 |
| 合成评分 | 2 条 fixture，strict EM=0.5；不是 benchmark 成绩 |
| 本机 HTTP fixture | 307 重定向不会转发带凭证的请求；没有请求真实模型服务 |
| 旧补丁 | 保留默认 dry-run、漂移/审阅哈希检查和备份回滚回归；本次未应用到原盘 |

覆盖范围包括工具参数与路径、禁止覆盖、FIFO 拒绝、模型/工具来源、调用 ID、写入授权、
步骤/工具/累积输入预算、严格指标与完整分母、索引、迁移、补丁审阅与回滚，以及 Jev/Qwen 分支。

真实 Jev/Qwen 端点、GPU smoke test、GRPO/GiGPO/ABC 训练、办公成功率与 Docker 构建尚未验收。
远程 CI 只验证下面列出的 CPU 软件行为，结果以 Actions 链接为准。

## 来源与环境

10 月 22 日 v0.2.0 交付包与当时的 MANIFEST 文件哈希一致；Mac 初次重跑 52 项通过。
本次恢复后在 Mac 工作副本重新实施修复并运行上述 72 项测试与安装检查。
原外接盘源码、数据、权重均未修改。原始快照无 Git 历史，无法给出其精确上游 commit。

## 原研究源码完整性

`legacy/Search-R1` 保留原主快照 260 文件，原 README 重定位；另有 8 个 AutoDL 脚本。
逐文件来源哈希见 `legacy/SOURCE_MANIFEST.json`。全仓库 224 个 Python 文件与 37 个 Shell 脚本通过语法检查；68 个维护文档相对链接通过。新增基线源码做文件一致性检查，
不导入 GPU 依赖，不将其视为功能验收。旧代码的已知问题单独列出。

## GitHub 同步

目标为 `ziyangyangg/Search_Agent_RL`。此前连接的 403 失败属于授权问题；现改用本机
已授权的 GitHub CLI 凭证通过 Git 完成推送。发布后已重新克隆并通过下述首次安装验收。
首次完整发布提交的 Linux Python 3.10/3.12 CI 已通过；后续提交状态可在 Actions 查看。

## 全新远程克隆验收

首个完整发布提交 `998aa511f614fefb52d6d31c732f60db2d459e05` 已成功推送，并从 GitHub
重新克隆到新目录。克隆 HEAD 与本地一致，322 个清单哈希和全部原始源码哈希通过；
在克隆副本中重跑 72 项测试通过，再构建 wheel、装入临时虚拟环境，并离开源码目录
验证 help、demo、index、score，均通过。

- [GitHub Actions 首次运行](https://github.com/ziyangyangg/Search_Agent_RL/actions/runs/40953340095)
- [所有 CI 运行](https://github.com/ziyangyangg/Search_Agent_RL/actions)

CI 覆盖 Ubuntu 上 Python 3.10/3.12 的文件完整性、软件回归、安装包和命令入口。
Windows 尚未实机验收；历史 GPU 训练与真实模型能力也不在这些检查的证明范围内。
