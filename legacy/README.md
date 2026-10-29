# 原始研究源码入口

本目录保留外接盘中实际存在的 Search-R1 本地快照与 AutoDL 脚本，便于研究训练流程和继续迁移。
默认应用的安装、测试与 CLI 仍从仓库根目录开始；此处的历史训练环境应单独配置。

| 路径 | 内容 |
| --- | --- |
| [Search-R1/README.md](Search-R1/README.md) | 本仓库提供的阅读、资产与运行边界指南 |
| [Search-R1/README.upstream.md](Search-R1/README.upstream.md) | 原快照自带说明，原样保留，仅作历史资料 |
| [Search-R1/search_r1](Search-R1/search_r1) | 搜索 rollout、检索服务和模型协议 |
| [Search-R1/verl](Search-R1/verl) | 原快照内的训练器、worker、模型适配和配置 |
| [Search-R1/eval/src](Search-R1/eval/src) | 本地独立评估、数据构建和 LoRA 实验 |
| [autodl-scripts](autodl-scripts) | 原盘旁路 AutoDL 脚本，包含与主目录略有差别的版本 |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | 原文件 SHA-256、README 重定位与来源边界 |

原始主快照 260 个文件均保留；原 `README.md` 改名为 `README.upstream.md`，其内容未改。
另保留 8 个旁路 AutoDL 脚本。原始目录没有 Git 历史，精确上游 commit 未知。
原文件保留许可证和作者声明，商业指南、原始数据、权重、空日志、缓存与安装产物不进入仓库。

此目录是待继续修复的历史研究基线，已知问题见[缺点清单](../docs/known-issues.md)。
新运行器的 72 项 CPU 测试不构成旧 GPU 训练器通过验收的证明。
