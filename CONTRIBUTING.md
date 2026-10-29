# 贡献指南

先读 [README](README.md)、[架构](docs/architecture.md) 与[已知问题](docs/known-issues.md)。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
agent-workbench demo --workspace examples/office
```

模型适配器只返回 `Completion`，工具经 `ToolSpec` 注册参数和读写效果，运行器统一校验/执行。不要让训练框架成为推理的必选依赖。注释解释来源边界、失败处理与设计原因，名称直接表达行为。

修协议、权限、预算、评分或安装行为时，提供能重现缺陷的回归；文字微调不用凑测试。PR 写明触发条件、改后行为、验证和仍未覆盖部分。保持 Python 3.10 兼容，新增依赖说明必要性。

评估固定预测与完整分母，记录模型、语料、模板、解码与预算。缺 token IDs/logprobs/provenance 的文本记录不能标为 trainer-ready。不要提交密钥、私有轨迹、数据、权重或商业指南，保留 LICENSE、NOTICE 与原始归属声明。
