# -*- coding: utf-8 -*-
"""
prompts.py
==========
存放「搜索智能体」的系统提示词（System Prompt）与对话模板。

设计参考 Search-R1 / R1-Searcher 范式：让模型在 <think></think> 里推理，
需要外部知识时用 <search>查询词</search> 调用检索工具，工具返回的资料会被放进
<information></information>，模型读到资料后继续推理，最终把答案写在 <answer></answer> 里。

注意：这是「纯文本协议」——我们不依赖任何框架的 function-calling，而是用一组特殊标签
来约定智能体与工具之间的交互。这样做的好处是：实现透明、可控、易于复现，并且训练时
可以精确地把「工具返回的 token」屏蔽掉（不参与梯度）。
"""

from __future__ import annotations

# 四类标签（与 Search-R1 源码一致）
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
SEARCH_OPEN, SEARCH_CLOSE = "<search>", "</search>"
INFO_OPEN, INFO_CLOSE = "<information>", "</information>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

# 生成时遇到这些子串就停下来：要么是发起了一次检索，要么是给出了最终答案
STOP_STRINGS = [SEARCH_CLOSE, ANSWER_CLOSE]

SYSTEM_PROMPT = (
    "你是一个善于通过『搜索-推理』来回答复杂问题的智能体。\n"
    "对于用户的问题，你必须严格遵循下面的交互协议：\n"
    "1. 每次拿到新信息后，先在 <think> 和 </think> 之间进行推理；\n"
    "2. 如果你发现还缺少知识，就调用搜索引擎：把检索关键词写在 <search> 和 </search> 之间，\n"
    "   系统会把检索到的资料返回在 <information> 和 </information> 之间；\n"
    "3. 你可以多次搜索（每次只搜一个最关键的子问题），直到你认为信息足够；\n"
    "4. 当且仅当信息足够时，把最终答案写在 <answer> 和 </answer> 之间，答案要简短精确，\n"
    "   例如 <answer> 北京 </answer>。不要在 <answer> 里写多余的解释。\n"
)

# 英文版系统提示（HotpotQA / 2Wiki / Bamboogle 都是英文数据集，训练时默认用英文版）
SYSTEM_PROMPT_EN = (
    "You are a research agent that answers complex questions by interleaving reasoning and search.\n"
    "You MUST follow this protocol every time:\n"
    "1. First reason inside <think> and </think> whenever you receive new information.\n"
    "2. If you lack knowledge, call the search engine by writing a query inside <search> and </search>; "
    "the system will return the top results inside <information> and </information>.\n"
    "3. You may search multiple times (one focused sub-question per search) until you have enough information.\n"
    "4. Only when the information is sufficient, give the final answer inside <answer> and </answer>. "
    "Keep the answer short and exact, e.g. <answer> Beijing </answer>. Do not put explanations inside <answer>.\n"
)


def build_user_turn(question: str) -> str:
    """把问题包装成用户回合的文本。"""
    return "问题(Question): " + question.strip()


def build_chat_messages(question: str, language: str = "en") -> list:
    """
    返回 OpenAI / HuggingFace chat 格式的消息列表，交给 tokenizer.apply_chat_template 使用。
    language='en' 用英文系统提示（默认，匹配英文 QA 数据集）；'zh' 用中文。
    """
    system = SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_turn(question)},
    ]


def apply_chat(tokenizer, messages, return_tensors="pt"):
    """
    统一渲染对话模板（兼容 Qwen2.5 与 Qwen3）。
    - 对 Qwen3 等带"思考模式"的模型传 enable_thinking=False，避免它自动产生 <think> 块、
      与本项目自定义的 <search>/<answer> 标签协议打架；
    - 对不支持该参数的模型(如 Qwen2.5)，apply_chat_template 会把多余关键字忽略，安全。
    （用 Qwen/Qwen3-4B-Instruct-2507 这类"非思考"检查点时本就不产生 <think>，传不传都行。）
    """
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors=return_tensors, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors=return_tensors)


def format_information_block(passages: list) -> str:
    """
    把检索到的若干段落格式化成一个 <information> 块，注入到对话中。
    passages: List[str]
    """
    body = "\n".join("(%d) %s" % (i + 1, p.strip()) for i, p in enumerate(passages))
    return "\n" + INFO_OPEN + "\n" + body + "\n" + INFO_CLOSE + "\n"
