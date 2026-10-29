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

真实 Jev/Qwen 端点、GPU smoke test、GRPO/GiGPO/ABC 训练、办公成功率、Docker 构建和
远程 GitHub Actions 结果尚未验收。CI 配置已准备，不能将配置文件当成远程运行成功。

## 来源与环境

10 月 22 日 v0.2.0 交付包与当时的 MANIFEST 文件哈希一致；Mac 初次重跑 52 项通过。
本次恢复后在 Mac 工作副本重新实施修复并运行上述 72 项测试与安装检查。
原外接盘源码、数据、权重均未修改。原始快照无 Git 历史，无法给出其精确上游 commit。

## GitHub 状态

目标为 `ziyangyangg/Search_Agent_RL`。账号元数据有 push 权限，但当前 GitHub 集成创建文件返回
`403 Resource not accessible by integration`，因此本次发布尚未写入目标仓库。
源码、文档与可上传交付包已经准备；需要重新授权连接访问该仓库并具备写入能力后再同步。
