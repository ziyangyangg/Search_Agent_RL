# -*- coding: utf-8 -*-
"""
retriever.py
============
检索工具（智能体调用的"搜索引擎"）。提供三种后端，按依赖从轻到重：

  1) SimpleBM25  —— 纯 Python 标准库实现的 BM25，**零依赖**、可在 CPU/任何机器跑，
                    用于 selfcheck 与冒烟测试，也足以在 MB 级小语料上训练跑通。
  2) bm25s       —— 高性能稀疏检索（pip install bm25s），无需 GPU，推荐做正式训练的默认后端。
  3) DenseRetriever —— 稠密检索：sentence-transformers 编码 + FAISS 近邻搜索，效果最好，
                    需要 GPU/faiss（pip install sentence-transformers faiss-cpu）。

为什么要"自建小语料"？标准 wiki-18 稠密索引高达 ~64GB，单卡复现不现实。
我们改为：用 HotpotQA distractor 自带的上下文段落构建一个 MB 级语料（见 build_dataset.py），
检索空间虽然受限于数据集，但完全够用来验证「智能体能否学会何时搜、搜得准不准」，
并且让整个项目在一张租来的卡上就能跑通。正式追求泛化时再换大语料即可。

统一接口：retriever.search(query: str, topk: int) -> List[str]
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9一-鿿]+")


def simple_tokenize(text: str):
    """小写化 + 抽取字母/数字/中文连续片段。够用且确定性强。"""
    return _TOKEN_RE.findall(text.lower())


def load_corpus(path: str):
    """
    读取 jsonl 语料，每行形如 {"id": "...", "contents": "..."}。
    返回 List[str]（contents 列表）。
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(obj.get("contents", obj.get("text", "")))
    return docs


# ----------------------------------------------------------------------------- #
# 1) 纯标准库 BM25
# ----------------------------------------------------------------------------- #
class SimpleBM25:
    """
    经典 BM25（Okapi）。参数 k1、b 取常用默认值。仅用标准库，便于离线验证与小语料训练。
    """

    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [simple_tokenize(d) for d in self.docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.N = max(1, len(self.docs))
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.doc_freqs = [Counter(t) for t in self.doc_tokens]
        # 文档频率 df：包含某词的文档数
        df = Counter()
        for toks in self.doc_tokens:
            for w in set(toks):
                df[w] += 1
        # idf（带 +0.5 平滑，保证非负）
        self.idf = {}
        for w, freq in df.items():
            self.idf[w] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def _score(self, q_tokens, idx: int) -> float:
        score = 0.0
        freqs = self.doc_freqs[idx]
        dl = self.doc_len[idx]
        for w in q_tokens:
            if w not in freqs:
                continue
            idf = self.idf.get(w, 0.0)
            tf = freqs[w]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
        return score

    def search(self, query: str, topk: int = 3):
        q_tokens = simple_tokenize(query)
        if not q_tokens:
            return []
        scored = [(self._score(q_tokens, i), i) for i in range(self.N)]
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, i in scored[:topk]:
            if s <= 0:
                continue
            results.append(self.docs[i])
        return results


# ----------------------------------------------------------------------------- #
# 2) bm25s 后端（高性能稀疏，无需 GPU）
# ----------------------------------------------------------------------------- #
class BM25sRetriever:
    def __init__(self, docs):
        import bm25s  # 延迟导入，缺包时不影响 SimpleBM25

        self.docs = list(docs)
        self._bm25s = bm25s
        self.retriever = bm25s.BM25(corpus=self.docs)
        tokens = bm25s.tokenize(self.docs, stopwords="en")
        self.retriever.index(tokens)

    def search(self, query: str, topk: int = 3):
        q_tokens = self._bm25s.tokenize([query], stopwords="en")
        results, scores = self.retriever.retrieve(q_tokens, k=min(topk, len(self.docs)))
        out = []
        for r in results[0]:
            out.append(r if isinstance(r, str) else self.docs[int(r)])
        return out


# ----------------------------------------------------------------------------- #
# 3) 稠密检索后端（sentence-transformers + FAISS）
# ----------------------------------------------------------------------------- #
class DenseRetriever:
    def __init__(self, docs, model_name: str = "intfloat/e5-base-v2", batch_size: int = 256,
                 cache_dir: str = None):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.faiss = faiss
        self.docs = list(docs)
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        # e5 系列要求给文档加 "passage: " 前缀、给查询加 "query: " 前缀
        self.is_e5 = "e5" in model_name.lower()
        passages = [("passage: " + d) if self.is_e5 else d for d in self.docs]
        emb = self.model.encode(passages, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=True, convert_to_numpy=True)
        emb = emb.astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])  # 归一化后内积=余弦相似度
        self.index.add(emb)

    def search(self, query: str, topk: int = 3):
        q = ("query: " + query) if self.is_e5 else query
        qe = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        topk = min(topk, len(self.docs))
        scores, idxs = self.index.search(qe, topk)
        return [self.docs[int(i)] for i in idxs[0] if i >= 0]


# ----------------------------------------------------------------------------- #
# 工厂函数
# ----------------------------------------------------------------------------- #
def load_retriever(corpus_path: str, backend: str = "bm25s", **kwargs):
    """
    backend: 'simple' | 'bm25s' | 'dense'
    缺少依赖时自动回退到 SimpleBM25，保证一定能跑（并打印提示）。
    """
    docs = load_corpus(corpus_path)
    if backend == "dense":
        try:
            return DenseRetriever(docs, **kwargs)
        except Exception as e:  # noqa
            print("[retriever] dense 后端不可用(%s)，回退到 SimpleBM25" % e)
            return SimpleBM25(docs)
    if backend == "bm25s":
        try:
            return BM25sRetriever(docs)
        except Exception as e:  # noqa
            print("[retriever] bm25s 不可用(%s)，回退到 SimpleBM25" % e)
            return SimpleBM25(docs)
    return SimpleBM25(docs)
