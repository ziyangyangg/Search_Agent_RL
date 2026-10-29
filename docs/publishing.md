# 发布与同步

目标仓库：<https://github.com/ziyangyangg/Search_Agent_RL>。

当前代码已完成本地软件验证。GitHub 连接读到了仓库与账号 push 权限，但创建文件返回
`403 Resource not accessible by integration`；这不是代码错误，也不代表文件已经上传。
需要在 GitHub/ChatGPT 的连接授权中检查该仓库是否被包含、是否允许写入，然后重新执行同步。

如选择在本机使用自己的 Git 身份发布，工作副本可按以下步骤提交；首次设置身份使用自己的信息：

```bash
git init -b main
git add .
git commit -m "Prepare audited Search Agent Workbench v0.2.1"
git remote add origin https://github.com/ziyangyangg/Search_Agent_RL.git
git push -u origin main
```

这些首次初始化命令适用于解压后的无 Git 目录，并假设目标仓库仍为空；已有 Git 工作副本复用现有历史和 remote。
本次整理的本机工作副本已经准备 Git 提交和 origin 时，只需检查 `git status` / `git remote -v` 后同步。
不要在命令、文档或仓库里粘贴 token；使用本机正常 Git 认证流程。

发布范围是新应用、测试、公开样例、说明和旧快照补丁；权重、原始数据、商业指南、运行轨迹及私密配置均不提交。
远程 Actions 启动后需查看实际结果，当前验证记录只声明已经运行过的本地检查。
