# -*- coding: utf-8 -*-
"""
rewards.py
==========
可验证奖励（RLVR, Reinforcement Learning from Verifiable Rewards）的实现。

核心思想：不训练一个「奖励模型」(像 RLHF 那样)，而是直接用一个**可程序化验证**的规则
来给智能体的回答打分——因为问答数据集自带标准答案(gold answer)，我们可以用
Exact Match / F1 这种确定性的指标来判断答案对不对。这样的奖励无偏、不会被模型"骗"
(reward hacking 风险低)，是 DeepSeek-R1、Search-R1 这一类工作能成立的关键。

本文件只依赖 Python 标准库，因此可以在任何机器上 `python3 selfcheck.py` 直接验证，
不需要 GPU、不需要安装 torch。
"""

from __future__ import annotations

import re
import string
from collections import Counter

from prompts import ANSWER_OPEN, ANSWER_CLOSE, SEARCH_OPEN, SEARCH_CLOSE, THINK_OPEN, THINK_CLOSE

# ----------------------------------------------------------------------------- #
# 1) 答案抽取：从智能体的完整轨迹里取出 <answer>...</answer> 中的内容
# ----------------------------------------------------------------------------- #
_ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL)
_SEARCH_RE = re.compile(re.escape(SEARCH_OPEN) + r"(.*?)" + re.escape(SEARCH_CLOSE), re.DOTALL)


def extract_answer(text: str):
    """返回轨迹中最后一个 <answer> 的内容；若没有则返回 None。"""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_searches(text: str):
    """返回轨迹中所有 <search> 查询词，按出现顺序。"""
    return [m.strip() for m in _SEARCH_RE.findall(text)]


def extract_last_query(text: str):
    """生成被 </search> 截断后，取最后一个未闭合/已闭合的查询词，供 rollout 调检索用。"""
    # 兼容两种情况：文本里已经含 </search>，或刚好以 </search> 结尾被截断
    searches = extract_searches(text)
    if searches:
        return searches[-1]
    # 没有闭合标签时，尝试取最后一个 <search> 之后的内容
    idx = text.rfind(SEARCH_OPEN)
    if idx != -1:
        return text[idx + len(SEARCH_OPEN):].replace(SEARCH_CLOSE, "").strip()
    return None


# ----------------------------------------------------------------------------- #
# 2) 文本归一化与 EM / F1（SQuAD / HotpotQA 官方口径）
# ----------------------------------------------------------------------------- #
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """SQuAD 风格归一化：小写、去标点、去冠词(a/an/the)、压缩空白。"""
    if s is None:
        return ""

    def remove_articles(text):
        return _ARTICLES_RE.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _as_gold_list(gold):
    """标准答案可能是单个字符串，也可能是别名列表(golden_answers)。统一成 list。"""
    if gold is None:
        return []
    if isinstance(gold, str):
        return [gold]
    return list(gold)


def exact_match(pred: str, gold) -> float:
    """归一化后完全相等则为 1.0，否则 0.0（gold 是别名列表时取最大）。"""
    npred = normalize_answer(pred)
    return 1.0 if any(npred == normalize_answer(g) for g in _as_gold_list(gold)) else 0.0


def cover_match(pred: str, gold) -> float:
    """Cover-EM / 子串匹配：标准答案（归一化后）是预测的子串则算命中。多跳问答常用。"""
    npred = normalize_answer(pred)
    for g in _as_gold_list(gold):
        ng = normalize_answer(g)
        if ng and ng in npred:
            return 1.0
    return 0.0


def f1_score(pred: str, gold) -> float:
    """token 级 F1（对多个 gold 取最大值）。"""
    best = 0.0
    pred_tokens = normalize_answer(pred).split()
    for g in _as_gold_list(gold):
        gold_tokens = normalize_answer(g).split()
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            # 两边都空才算对，否则 0
            best = max(best, 1.0 if pred_tokens == gold_tokens else 0.0)
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


# ----------------------------------------------------------------------------- #
# 3) 格式奖励：鼓励模型遵守 <think>/<search>/<answer> 协议
# ----------------------------------------------------------------------------- #
def format_reward(text: str) -> float:
    """
    简单的结构检查，返回 0~1：
      - 含有成对的 <think></think>          +0.34
      - 含有 <answer></answer>             +0.33
      - 标签基本配平（开/闭数量一致）        +0.33
    不追求严格状态机，避免奖励过度工程化(over-engineering 反而招致 reward hacking)。
    """
    score = 0.0
    if THINK_OPEN in text and THINK_CLOSE in text:
        score += 0.34
    if ANSWER_OPEN in text and ANSWER_CLOSE in text:
        score += 0.33
    balanced = (
        text.count(THINK_OPEN) == text.count(THINK_CLOSE)
        and text.count(ANSWER_OPEN) == text.count(ANSWER_CLOSE)
        and text.count(SEARCH_OPEN) == text.count(SEARCH_CLOSE)
    )
    if balanced:
        score += 0.33
    return round(score, 4)


# ----------------------------------------------------------------------------- #
# 4) 组合奖励：训练时真正用到的标量奖励
# ----------------------------------------------------------------------------- #
def compute_reward(
    text: str,
    gold,
    num_searches: int,
    answer_metric: str = "f1",
    w_answer: float = 1.0,
    w_format: float = 0.2,
    w_search: float = 0.1,
    max_searches_penalty_after: int = 4,
    length_penalty_after: int = 5,
    length_penalty_per_word: float = 0.0,
    no_search_penalty: float = 0.0,
):
    """
    计算单条轨迹的总奖励，并返回各分项（便于日志和调参）。

    参数
    ----
    text            : 智能体生成的完整轨迹（含 think/search/information/answer）
    gold            : 标准答案（str 或 别名 list）
    num_searches    : 该轨迹实际发起的检索次数
    answer_metric   : 'f1' | 'em' | 'cover' —— 答案分用哪种指标
                      （小模型/小语料推荐 'f1'，信号更平滑，更利于 RL 收敛）
    w_answer/w_format/w_search : 各分项权重
    max_searches_penalty_after : 搜索次数超过该值后，每多一次轻微扣分（防止无意义刷搜索）

    返回
    ----
    (reward: float, info: dict)
    """
    pred = extract_answer(text)
    has_answer = pred is not None

    if answer_metric == "em":
        ans = exact_match(pred or "", gold)
    elif answer_metric == "cover":
        ans = cover_match(pred or "", gold)
    else:
        ans = f1_score(pred or "", gold)

    fmt = format_reward(text)

    # 鼓励"至少搜一次"（多跳问题闭门造车通常答不对），但不鼓励无脑刷搜索
    if num_searches >= 1:
        search_bonus = 1.0
    else:
        search_bonus = 0.0
    over = max(0, num_searches - max_searches_penalty_after)
    search_term = search_bonus - 0.1 * over

    # v3.1 新增：length penalty（鼓励短答） + no_search hard penalty
    length_pen = 0.0
    if has_answer and pred:
        words = pred.split()
        if len(words) > length_penalty_after:
            excess = min(len(words) - length_penalty_after, 15)
            length_pen = -length_penalty_per_word * excess
    no_search_term = -no_search_penalty if num_searches == 0 else 0.0
    reward = (w_answer * ans + w_format * fmt + w_search * search_term
              + length_pen + no_search_term)

    info = {
        "reward": round(reward, 4),
        "answer_score": round(ans, 4),
        "format": fmt,
        "num_searches": num_searches,
        "has_answer": has_answer,
        "pred": pred,
        "length_pen": round(length_pen, 4),
        "no_search_pen": round(no_search_term, 4),
    }
    return reward, info
