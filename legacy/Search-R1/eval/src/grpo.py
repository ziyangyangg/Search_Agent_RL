# -*- coding: utf-8 -*-
"""
grpo.py
=======
GRPO（Group Relative Policy Optimization，组相对策略优化）的核心数学。

GRPO 是 DeepSeekMath 提出、被 DeepSeek-R1 发扬光大的 RL 算法。相对 PPO 的最大改动：
**去掉了价值网络(critic)**，改用「同一个问题采样一组(G 条)回答，用组内奖励的均值做基线」
来估计优势(advantage)。省掉一个和策略同样大的 critic，显存大约省一半，特别适合单卡复现。

优势计算（本文件 group_normalized_advantages，纯标准库，可离线验证）：
    对每个问题 q，采样 G 条回答，奖励为 r_1..r_G
    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
直观理解：比组内平均好的回答，优势为正(鼓励)；比平均差的，优势为负(抑制)。
若一组里所有回答奖励都一样(std=0)，优势全为 0——这组学不到东西，正常。

损失（masked_grpo_loss，需要 torch）：
    L = - (1/Σmask) Σ_t mask_t * A * logπ_θ(token_t | <t)   [+ β * KL(π_θ || π_ref)]
其中 mask 只在「模型自己生成的 token」上为 1，prompt 和 <information> 注入的 token 为 0。
当每批 rollout 只做一次梯度更新(num_iterations=1, on-policy)时，重要性比率恒为 1，
GRPO 退化为「带组基线的 REINFORCE」——这正是很多开源迷你实现采用的稳健写法。
我们默认走这条路（简单、不易出错），同时提供带 clip 比率的完整版以供学习。
"""

from __future__ import annotations

from statistics import mean, pstdev


def group_normalized_advantages(rewards, group_size, eps=1e-6):
    """
    rewards    : 扁平列表，按 [q0_g0, q0_g1, ..., q0_g{G-1}, q1_g0, ...] 排列
    group_size : G
    返回与 rewards 等长的优势列表。
    """
    assert len(rewards) % group_size == 0, "rewards 数量必须是 group_size 的整数倍"
    advantages = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start:start + group_size]
        mu = mean(group)
        sigma = pstdev(group)  # 总体标准差
        for r in group:
            advantages.append((r - mu) / (sigma + eps))
    return advantages


def per_token_logprobs(logits, input_ids):
    """
    计算每个位置上「真实下一个 token」的 log 概率（torch 实现，训练时用）。

    logits    : [B, T, V]，模型对每个位置预测下一个 token 的分布
    input_ids : [B, T]
    返回 logp : [B, T-1]，logp[b, t] = log P(input_ids[b, t+1] | input_ids[b, :t+1])
    """
    import torch
    import torch.nn.functional as F

    # 用前 T-1 个位置的 logits 预测后 T-1 个 token
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp_all = F.log_softmax(logits.float(), dim=-1)
    logp = torch.gather(logp_all, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return logp  # [B, T-1]


def masked_grpo_loss(policy_logits, input_ids, loss_mask, advantages,
                     ref_logits=None, beta=0.0,
                     old_logp=None, clip_eps=0.2):
    """
    计算一个 batch 的 GRPO 损失（torch）。

    policy_logits : [B, T, V] 当前策略模型的 logits
    input_ids     : [B, T]
    loss_mask     : [B, T]，1=模型生成的 token（参与损失），0=prompt/注入资料（屏蔽）
                    注意 mask 与 token 对齐；内部会自动错位成对「下一个 token」的 mask。
    advantages    : [B]，每条轨迹一个标量优势（broadcast 到该轨迹所有 token）
    ref_logits    : [B, T, V] 参考模型 logits；beta>0 时用于 KL 正则；beta=0 时不需要(省显存)
    beta          : KL 系数。默认 0.0 → 不加载参考模型（现代实践常见做法）
    old_logp      : [B, T-1] 采样时的旧 log 概率；提供则启用 PPO 式 clip 比率，否则走 REINFORCE 路径
    clip_eps      : clip 比率的 ε

    返回 (loss, metrics_dict)
    """
    import torch

    logp = per_token_logprobs(policy_logits, input_ids)      # [B, T-1]
    # mask 对齐到「被预测的 token」(即 input_ids[:,1:])
    mask = loss_mask[:, 1:].float()                          # [B, T-1]
    adv = advantages.view(-1, 1)                             # [B, 1]

    if old_logp is not None:
        # 完整 GRPO：重要性比率 + clip
        ratio = torch.exp(logp - old_logp)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        pg = -torch.min(unclipped, clipped)                 # 取较小者→保守更新
    else:
        # on-policy(μ=1) 路径：比率=1，退化为带组基线的策略梯度
        pg = -(adv * logp)

    # KL 正则（可选）：把策略拉住别离参考模型太远
    kl_term = torch.zeros_like(pg)
    if beta and beta > 0.0 and ref_logits is not None:
        ref_logp = per_token_logprobs(ref_logits, input_ids).detach()
        # k3 估计量（无偏、非负）：exp(Δ) - Δ - 1，Δ = ref_logp - logp
        delta = ref_logp - logp
        kl_term = torch.exp(delta) - delta - 1.0
        per_token = pg + beta * kl_term
    else:
        per_token = pg

    denom = mask.sum().clamp(min=1.0)
    loss = (per_token * mask).sum() / denom

    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach().cpu()),
            "mean_logp": float((logp * mask).sum().div(denom).detach().cpu()),
            "mean_adv": float(advantages.mean().detach().cpu()),
            "kl": float((kl_term * mask).sum().div(denom).detach().cpu()) if (beta and ref_logits is not None) else 0.0,
            "n_tokens": int(mask.sum().item()),
        }
    return loss, metrics
