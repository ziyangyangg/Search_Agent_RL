#!/bin/bash
set -e
cd /root/Search-R1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl_env
export JAVA_HOME=$CONDA_PREFIX/lib/jvm
export JVM_PATH=$CONDA_PREFIX/lib/server/libjvm.so
unset http_proxy https_proxy
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/server:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo DEBUG_JH=$JAVA_HOME; echo DEBUG_LIBJVM=$(ls $JAVA_HOME/lib/server/libjvm.so 2>&1); which python
python search_r1/search/retrieval_server.py --index_path /root/Search-R1/data/bm25_index/bm25 --corpus_path /root/Search-R1/data/corpus/wiki-18.jsonl --topk 3 --retriever_name bm25
