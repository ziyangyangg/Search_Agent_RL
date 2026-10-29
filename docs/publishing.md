# 发布与同步

目标仓库：<https://github.com/ziyangyangg/Search_Agent_RL>。

本仓库默认运行 Workbench 应用；原 Search-R1 研究源码保留在独立 `legacy/` 中。
发布包括源码、测试、说明、公开样例和来源记录；原始数据、权重、商业指南和私有轨迹不提交。

本机 GitHub CLI 授权示例：

```bash
gh auth login --hostname github.com --git-protocol https --web --scopes workflow
gh auth setup-git --hostname github.com
gh auth status
```

`workflow` 用于提交本仓库的 `.github/workflows/`。不要将 token 写入命令、文件或仓库。
ChatGPT GitHub 连接和本机 CLI 是不同授权渠道；此前连接 403 时，改用已授权本机 Git。

新贡献者拉取并修改：

```bash
git clone https://github.com/ziyangyangg/Search_Agent_RL.git
cd Search_Agent_RL
git switch -c improve/my-change
python -m pip install -e .
python -m unittest discover -s tests -v
```

完成修改后提交到分支并发 PR。维护者应核对测试与已知问题，再更新主分支。所有来源哈希和根目录
`MANIFEST.json` 应随有意的源码变更更新；原始基线修改还需记录在来源说明中。
远程 Actions 以实际运行结果为准，不能只凭配置存在宣称已通过。
