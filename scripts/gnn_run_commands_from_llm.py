#!/usr/bin/env python3
"""Generate GNN run commands from LLM text-extraction configs."""

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from glam4cm.run_configs import (
    EDGE_ATTR_GNN_MODELS,
    GNN_ONLY_ARGS,
    HEADED_GNN_MODELS,
    LLM_TO_GNN_TASK,
    command_params,
    get_task_id,
    infer_embedding_checkpoint,
    model_dir_for_tokens,
    normalize_config_line,
    remove_args,
    set_arg,
    split_option,
)


TEXT_FLAG_ALIASES = {
    "use_node_attributes": "use_attributes",
}

LLM_ONLY_ARGS = {
    "eval_batch_size",
    "freeze_pretrained_weights",
    "model_name",
    "num_eval_steps",
    "num_log_steps",
    "num_save_steps",
    "train_batch_size",
    "warmup_steps",
}

EMBEDDING_ARGS = {
    "ckpt",
    "use_embeddings",
    "use_embedding",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read LLM text-extraction configs and print matching "
            "python -m glam4cm.run commands for GNN tasks."
        )
    )
    parser.add_argument("configs_file", help="Text, CSV, or JSON file containing LLM configs.")
    parser.add_argument("--output", "-o", help="Optional file to write generated commands.")
    parser.add_argument("--tokens_only", action="store_true", help="Print only run.py args.")
    parser.add_argument("--python", default="python", help="Python executable used in generated commands.")

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gnn_conv_model", nargs="+", default=["SAGEConv", "GATv2Conv"])
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_conv_layers", type=int, default=3)
    parser.add_argument("--num_mlp_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aggregation", default="sum")
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--neg_sampling_ratio", type=int, default=1)
    parser.add_argument("--models_dir", default=None)

    parser.add_argument(
        "--require_embeddings",
        action="store_true",
        help="Generate only configs with --use_embeddings.",
    )
    parser.add_argument(
        "--require_edge_attrs",
        action="store_true",
        help="Generate only configs with --use_edge_attrs.",
    )
    parser.add_argument(
        "--no_embeddings",
        action="store_true",
        help="Do not generate --use_embeddings configs.",
    )
    parser.add_argument(
        "--no_edge_attrs",
        action="store_true",
        help="Do not generate --use_edge_attrs configs.",
    )
    parser.add_argument("--l_norm", action="store_true")
    parser.add_argument("--bias", action="store_true")
    return parser.parse_args()


def normalize_token_aliases(tokens: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for token in tokens:
        if not token.startswith("--"):
            normalized.append(token)
            continue
        key, value = split_option(token)
        key = TEXT_FLAG_ALIASES.get(key, key)
        normalized.append(f"--{key}" if value is None else f"--{key}={value}")
    return normalized


def params_to_tokens(params: Dict[str, object]) -> List[str]:
    tokens: List[str] = []
    for raw_key, value in params.items():
        key = TEXT_FLAG_ALIASES.get(str(raw_key).lstrip("-"), str(raw_key).lstrip("-"))
        if isinstance(value, bool):
            if value:
                tokens.append(f"--{key}")
        elif value is not None:
            tokens.append(f"--{key}={value}")
    return normalize_config_line(" ".join(shlex.quote(token) for token in tokens))


def read_json_configs(path: Path) -> List[List[str]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("configs", [payload])
    if not isinstance(payload, list):
        raise ValueError("JSON config must be an object, list, or {'configs': [...]} object.")
    configs = []
    for item in payload:
        if isinstance(item, str):
            configs.append(normalize_token_aliases(normalize_config_line(item)))
        elif isinstance(item, dict):
            configs.append(normalize_token_aliases(params_to_tokens(item)))
        else:
            raise ValueError(f"Unsupported JSON config item: {item!r}")
    return configs


def config_from_csv_row(row: Dict[str, str]) -> str:
    for column in ("Config", "config", "command", "final_command"):
        if row.get(column):
            return row[column]
    first_value = next((value for value in row.values() if value), "")
    return first_value


def read_text_or_csv_configs(path: Path) -> List[List[str]]:
    lines = path.read_text().splitlines()
    if not lines:
        return []

    configs: List[List[str]] = []
    first_line = lines[0].strip()
    if "," in first_line and not first_line.startswith("--"):
        reader = csv.DictReader(lines)
        for row in reader:
            line = config_from_csv_row(row).strip()
            if line and not line.startswith("#"):
                configs.append(normalize_token_aliases(normalize_config_line(line)))
        return configs

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            configs.append(normalize_token_aliases(normalize_config_line(stripped)))
    return configs


def read_configs(path: str) -> List[List[str]]:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        return read_json_configs(config_path)
    return read_text_or_csv_configs(config_path)


def gnn_flag_combinations(args: argparse.Namespace) -> Iterable[Dict[str, bool]]:
    use_embeddings_values = [False, True]
    use_edge_attrs_values = [False, True]
    if args.require_embeddings:
        use_embeddings_values = [True]
    elif args.no_embeddings:
        use_embeddings_values = [False]
    if args.require_edge_attrs:
        use_edge_attrs_values = [True]
    elif args.no_edge_attrs:
        use_edge_attrs_values = [False]

    for use_embeddings in use_embeddings_values:
        for use_edge_attrs in use_edge_attrs_values:
            yield {
                "use_embeddings": use_embeddings,
                "use_edge_attrs": use_edge_attrs,
            }


def base_gnn_tokens(args: argparse.Namespace, task_id: int, model_name: str) -> List[str]:
    tokens = [
        f"--task_id={LLM_TO_GNN_TASK[task_id]}",
        f"--num_epochs={args.num_epochs}",
        f"--lr={args.lr}",
        f"--batch_size={args.batch_size}",
        f"--gnn_conv_model={model_name}",
        f"--num_conv_layers={args.num_conv_layers}",
        f"--num_mlp_layers={args.num_mlp_layers}",
        f"--hidden_dim={args.hidden_dim}",
        f"--output_dim={args.output_dim}",
        f"--dropout={args.dropout}",
        f"--aggregation={args.aggregation}",
        f"--embed_batch_size={args.embed_batch_size}",
    ]
    if LLM_TO_GNN_TASK[task_id] == 8:
        tokens.append(f"--neg_sampling_ratio={args.neg_sampling_ratio}")
    if args.num_heads is not None:
        tokens.append(f"--num_heads={args.num_heads}")
    elif model_name in HEADED_GNN_MODELS:
        tokens.append("--num_heads=4")
    if args.l_norm:
        tokens.append("--l_norm")
    if args.bias:
        tokens.append("--bias")
    return tokens


def common_llm_tokens(llm_tokens: List[str]) -> List[str]:
    return remove_args(
        llm_tokens,
        {"task_id", "num_epochs", "lr"} | LLM_ONLY_ARGS | GNN_ONLY_ARGS | EMBEDDING_ARGS,
    )


def generated_tokens_for_llm(args: argparse.Namespace, llm_tokens: List[str]) -> List[List[str]]:
    llm_task_id = get_task_id(llm_tokens)
    if llm_task_id not in LLM_TO_GNN_TASK:
        return []

    params = command_params(llm_tokens)
    models_dir = args.models_dir or str(params.get("models_dir", "ftlm"))
    shared_tokens = common_llm_tokens(llm_tokens)
    if args.models_dir:
        shared_tokens = set_arg(shared_tokens, "models_dir", args.models_dir)
        llm_tokens = set_arg(llm_tokens, "models_dir", args.models_dir)

    generated: List[List[str]] = []
    for model_name in args.gnn_conv_model:
        for flags in gnn_flag_combinations(args):
            if flags["use_edge_attrs"] and model_name not in EDGE_ATTR_GNN_MODELS:
                continue
            tokens = base_gnn_tokens(args, llm_task_id, model_name)
            if flags["use_edge_attrs"]:
                tokens.append("--use_edge_attrs")
            if flags["use_embeddings"]:
                tokens.append("--use_embeddings")
                tokens.append(f"--ckpt={model_dir_for_tokens(llm_tokens, models_dir)}")
            tokens.extend(shared_tokens)
            generated.append(infer_embedding_checkpoint(tokens, models_dir))
    return generated


def shell_command(tokens: Sequence[str], args: argparse.Namespace) -> str:
    if args.tokens_only:
        parts = list(tokens)
    else:
        parts = [args.python, "-m", "glam4cm.run", *tokens]
    return " ".join(shlex.quote(part) for part in parts)


def main() -> int:
    args = parse_args()
    configs = read_configs(args.configs_file)
    commands: List[str] = []
    seen = set()

    for llm_tokens in configs:
        for tokens in generated_tokens_for_llm(args, llm_tokens):
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            commands.append(shell_command(tokens, args))

    output = "\n".join(commands)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + ("\n" if output else ""))
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
