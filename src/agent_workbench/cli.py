"""CLI entry points. Demo, real inference, and metrics are explicitly distinct."""

import argparse
import json
import sys
from pathlib import Path

from .evaluation import score_rows, sha256
from .models import ChatCompletionsBackend, ScriptedBackend
from .retrieval import CorpusSearch, build_index
from .runtime import Agent, RunConfig
from .tools import ToolRegistry, WorkspaceTools
from .types import Completion, ToolCall


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace an evaluation result or private trace.
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agent-workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="offline fixture; not a model evaluation")
    demo.add_argument("--workspace", type=Path, required=True)
    demo.add_argument("--trace", type=Path)
    run = commands.add_parser("run", help="execute with a real compatible tool-calling model")
    run.add_argument("task")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--base-url", required=True, help="versioned endpoint, e.g. http://127.0.0.1:8000/v1")
    run.add_argument("--api-key-env", default="AGENT_API_KEY")
    run.add_argument("--corpus-index", type=Path)
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-tool-calls", type=int, default=12)
    run.add_argument("--max-context-chars", type=int, default=120000,
                     help="serialized input character ceiling; not a model token-window guarantee")
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    run.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"))
    run.add_argument("--jev-model", help="enable Jev candidate selection with this model ID")
    run.add_argument("--candidates-file", type=Path, help="JSON array of prepared read actions")
    run.add_argument("--fast-threshold", type=float, default=0.9,
                     help="experimental selected-option probability threshold, not a correctness guarantee")
    run.add_argument("--allow-workspace-writes", action="store_true",
                     help="authorize create-only files inside this workspace for this run")
    run.add_argument("--trace", type=Path, required=True)
    index = commands.add_parser("index", help="build a new SQLite FTS5 small-corpus index")
    index.add_argument("--corpus", type=Path, required=True)
    index.add_argument("--out", type=Path, required=True)
    score = commands.add_parser("score", help="score JSONL id/prediction/gold using strict metrics")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="incremental strict QA benchmark with a real endpoint")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--out-dir", type=Path, required=True)
    evaluate.add_argument("--corpus-index", type=Path)
    evaluate.add_argument("--modes", default="direct,rag,agent")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--base-url", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--api-key-env", default="AGENT_API_KEY")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            from .benchmark import evaluate_dataset
            backend = ChatCompletionsBackend(args.base_url, args.model, api_key_env=args.api_key_env)
            result = evaluate_dataset(backend, args.dataset, args.out_dir, index=args.corpus_index,
                                      modes=tuple(args.modes.split(",")), limit=args.limit,
                                      metadata={"model": args.model, "temperature": 0, "max_tokens": 1024})
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "index":
            print(json.dumps({"documents": build_index(args.corpus, args.out),
                              "backend": "sqlite_fts5", "corpus_sha256": sha256(args.corpus)}))
            return 0
        if args.command == "score":
            with args.predictions.open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            result = score_rows(rows)
            result["predictions_sha256"] = sha256(args.predictions)
            write_json(args.out, result)
            print(json.dumps({key: value for key, value in result.items() if key != "details"}))
            return 0
        registry = ToolRegistry()
        WorkspaceTools(args.workspace).register(registry)
        if args.command == "demo":
            backend = ScriptedBackend([
                Completion(tool_calls=(ToolCall("demo-list", "workspace_list_files", {}),), finish_reason="tool_calls"),
                Completion("Offline fixture finished. The workspace listing is recorded in the trace.")])
            trace = Agent(backend, registry).run("List workspace files.", metadata={"backend": "scripted_fixture", "capability_evaluation": False})
        else:
            if args.trace.exists():
                raise FileExistsError("trace path exists; choose a new path")
            if args.corpus_index:
                CorpusSearch(args.corpus_index).register(registry)
            backend = ChatCompletionsBackend(args.base_url, args.model,
                                            api_key_env=args.api_key_env,
                                            max_tokens=args.max_tokens,
                                            enable_thinking={"default": None, "on": True, "off": False}[args.thinking],
                                            reasoning_effort=args.reasoning_effort)
            if bool(args.jev_model) != bool(args.candidates_file):
                raise ValueError("--jev-model and --candidates-file must be supplied together")
            if args.jev_model:
                from .jev import JevDecisionClient
                from .cooperation import ActionCandidate, CooperativeBackend
                with args.candidates_file.open(encoding="utf-8") as source:
                    candidates = [ActionCandidate(**row) for row in json.load(source)]
                backend = CooperativeBackend(backend, JevDecisionClient(args.jev_model), candidates,
                                             min_probability=args.fast_threshold)
            config = RunConfig(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls,
                               max_context_chars=args.max_context_chars,
                               allow_workspace_writes=args.allow_workspace_writes)
            trace = Agent(backend, registry, config).run(args.task, metadata={
                "backend": "chat_completions", "model": args.model,
                "decoding": {"temperature": 0, "max_tokens": args.max_tokens,
                             "thinking": args.thinking, "reasoning_effort": args.reasoning_effort},
                "max_steps": args.max_steps, "max_tool_calls": args.max_tool_calls,
                "max_context_chars": args.max_context_chars,
                "workspace_writes_authorized": args.allow_workspace_writes})
            if args.jev_model:
                trace.metadata["cooperation"] = {"threshold": args.fast_threshold,
                                               "candidates_sha256": sha256(args.candidates_file),
                                               "routing_events": backend.routing_events}
        if args.trace:
            write_json(args.trace, trace.to_dict())
        print(json.dumps({"status": trace.status, "answer": trace.final_answer,
                          "usage": trace.usage}, ensure_ascii=False, indent=2))
        return 0 if trace.status == "completed" else 2
    except (OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
