# -*- coding: utf-8 -*-
"""
agent.py
========
「搜索智能体」的多轮 rollout（轨迹生成）驱动。

一条轨迹长这样（模型生成的部分 + 工具注入的部分交替）：

    <think> 我需要先查一下 X 是谁 </think>
    <search> who is X </search>
    <information>           <-- 这一段是检索工具注入的，不是模型生成的！训练时要 mask 掉
    (1) X is ...
    </information>
    <think> 现在我知道了 X，再查 Y </think>
    <search> ... </search>
    <information> ... </information>
    <think> 信息够了 </think>
    <answer> 最终答案 </answer>

为了既能离线测试、又能在 GPU 上训练，本文件把「LLM 生成」抽象成一个可注入的回调
`generate_fn(prompt_text) -> continuation_text`：
  - selfcheck / 单元测试里传入一个"剧本化"的假 LLM；
  - 评估时传入包装了 model.generate（或 vLLM）的真实回调；
  - 训练时（train_grpo_agent.py）用的是 token 级循环，但复用本文件的 `parse_action` 决策逻辑，
    保证「何时停、是搜索还是回答」这套状态机只有一处实现。
"""

from __future__ import annotations

from prompts import (
    SEARCH_OPEN, SEARCH_CLOSE, ANSWER_OPEN, ANSWER_CLOSE,
    format_information_block,
)
from rewards import extract_last_query, extract_answer


def parse_action(generated_text: str):
    """
    判断模型这一回合做了什么动作（状态机的核心，训练/评估共用）。

    返回 (action, payload)：
      ("answer", answer_str) —— 出现了 </answer>，轨迹结束
      ("search", query_str)  —— 出现了 </search>，需要调用检索
      ("none",   None)       —— 既没搜也没答（模型自己停了/超长），轨迹结束
    约定：若同一段里 </answer> 和 </search> 都出现，以**先出现**的为准。
    """
    i_ans = generated_text.find(ANSWER_CLOSE)
    i_sea = generated_text.find(SEARCH_CLOSE)

    has_ans = i_ans != -1
    has_sea = i_sea != -1

    if has_ans and (not has_sea or i_ans < i_sea):
        return "answer", (extract_answer(generated_text) or "")
    if has_sea:
        return "search", (extract_last_query(generated_text) or "")
    return "none", None


def run_agent_text(question, generate_fn, retrieve_fn, max_turns=4, topk=3, language="en"):
    """
    文本级 rollout（用于评估 / selfcheck；不涉及梯度）。

    参数
    ----
    question    : 问题字符串
    generate_fn : callable(prompt_text:str) -> str
                  给定到目前为止的完整对话文本，返回模型这一回合的"续写"（应包含停止标签）。
    retrieve_fn : callable(query:str, topk:int) -> List[str]
                  检索回调（见 retriever.py）。
    max_turns   : 最多允许多少个「生成回合」（一次搜索算一个回合）。
    topk        : 每次检索返回多少段。

    返回 dict：
      full_text     : 不含初始 system/user 提示的"智能体轨迹"文本（用于算奖励/抽答案）
      segments      : List[(text, is_model_generated)]，用于需要 mask 的场景
      num_searches  : 实际检索次数
      finished      : 是否以 <answer> 正常结束
      prompt_text   : 初始提示（system+user 渲染后的纯文本，仅用于把上下文喂给 generate_fn）
    """
    from prompts import build_chat_messages, build_user_turn, SYSTEM_PROMPT_EN, SYSTEM_PROMPT

    system = SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT
    prompt_text = system + "\n\n" + build_user_turn(question) + "\n"

    segments = []          # [(text, is_model_generated)]
    context = prompt_text  # 累积喂给 generate_fn 的完整上下文
    num_searches = 0
    finished = False

    for _turn in range(max_turns):
        continuation = generate_fn(context)
        if continuation is None:
            continuation = ""
        segments.append((continuation, True))   # 模型生成 → 训练时参与梯度
        context += continuation

        action, payload = parse_action(continuation)
        if action == "answer":
            finished = True
            break
        elif action == "search":
            num_searches += 1
            passages = retrieve_fn(payload, topk) if payload else []
            info_block = format_information_block(passages)
            segments.append((info_block, False))  # 工具注入 → 训练时 mask 掉
            context += info_block
            # 进入下一回合继续生成
        else:
            # 模型既没搜也没答，结束（奖励函数会因为没有 <answer> 而给低分）
            break

    full_text = "".join(t for t, _ in segments)
    return {
        "full_text": full_text,
        "segments": segments,
        "num_searches": num_searches,
        "finished": finished,
        "prompt_text": prompt_text,
    }


def naive_rag_text(question, generate_fn, retrieve_fn, topk=3, language="en"):
    """
    朴素 RAG 基线（对照组）：**只检索一次**，把问题原文当查询，检索 topk 段后直接让模型作答。
    用来和我们训练出来的「会多轮搜索的智能体」做 before/after 对比。
    """
    from prompts import build_user_turn, SYSTEM_PROMPT_EN, SYSTEM_PROMPT

    system = SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT
    passages = retrieve_fn(question, topk)
    info_block = format_information_block(passages)
    context = (system + "\n\n" + build_user_turn(question) + "\n" + info_block)
    continuation = generate_fn(context) or ""
    return {
        "full_text": info_block + continuation,
        "num_searches": 1,
        "finished": ANSWER_CLOSE in continuation,
        "prompt_text": context,
    }
