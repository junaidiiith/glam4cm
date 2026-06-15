import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm

from glam4cm.downstream_tasks.common_args import get_config_hash
from glam4cm.downstream_tasks.common_args import get_config_str
from glam4cm.downstream_tasks.utils import RESULTS_FILE_NAME
from glam4cm.run import tasks, tasks_handler_map
from glam4cm.settings import EDGE_CLS_TASK, GRAPH_CLS_TASK, LINK_PRED_TASK, NODE_CLS_TASK
from glam4cm.utils import md5_hash


LLM_TO_GNN_TASK = {
    2: 6,
    3: 7,
    4: 8,
    5: 9,
}
GNN_TO_LLM_TASK = {gnn: llm for llm, gnn in LLM_TO_GNN_TASK.items()}

TASK_NAMES = {
    2: f"LM_{GRAPH_CLS_TASK}",
    3: f"LM_{NODE_CLS_TASK}",
    4: f"LM_{LINK_PRED_TASK}",
    5: f"LM_{EDGE_CLS_TASK}",
    6: f"GNN_{GRAPH_CLS_TASK}",
    7: f"GNN_{NODE_CLS_TASK}",
    8: f"GNN_{LINK_PRED_TASK}",
    9: f"GNN_{EDGE_CLS_TASK}",
}

CONFIG_FLAGS = [
    "use_node_types",
    "use_attributes",
    "use_edge_label",
    "use_edge_types",
]

GNN_CONFIG_FLAGS = [
    "use_edge_attrs",
    "use_embeddings",
]

DATASET_IGNORED_CONFIG_FLAGS = {
    "eamodelset": {
        "use_attributes",
        "use_edge_label",
    },
    "ontouml": {
        "use_edge_label",
        "use_edge_types",
    },
}

EDGE_ATTR_GNN_MODELS = {
    "GATConv",
    "GATv2Conv",
    "TransformerConv",
}

HEADED_GNN_MODELS = {
    "GATConv",
    "GATv2Conv",
}

GNN_ONLY_ARGS = {
    "aggregation",
    "bias",
    "gnn_conv_model",
    "hidden_dim",
    "input_dim",
    "l_norm",
    "num_conv_layers",
    "num_heads",
    "num_mlp_layers",
    "output_dim",
    "residual",
    "use_edge_attrs",
}

EMBEDDING_ONLY_ARGS = {
    "ckpt",
    "embed_batch_size",
    "embed_model_name",
    "random_embed_dim",
    "randomize_ee",
    "randomize_ne",
    "regen_embeddings",
    "use_embeddings",
}

OPTION_ALIASES = {
    "use_embedding": "use_embeddings",
    "use_attrs": "use_edge_attrs",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run GLAM4CM configurations from either a text file or an "
            "argument-generated LLM/GNN configuration matrix."
        )
    )
    parser.add_argument("--configs_file", type=str, default=None)
    parser.add_argument("--status_file", type=str, default="logs/run_configurations_status.csv")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun", action="store_true")

    parser.add_argument("--cuda_devices", nargs="+", type=str, default=["0"])
    parser.add_argument("--run_llm", action="store_true")
    parser.add_argument("--run_gnn", action="store_true")

    parser.add_argument("--task_id", type=int, choices=sorted(LLM_TO_GNN_TASK), default=None)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["ecore_555"],
        choices=["modelset", "ecore_555", "mar-ecore-github", "eamodelset", "ontouml"],
    )
    parser.add_argument("--remove_duplicates", action="store_true")
    parser.add_argument("--include_dummies", action="store_true")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--min_enr", type=float, default=-1.0)
    parser.add_argument("--min_edges", type=int, default=-1)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--node_topk", type=int, default=-1)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--models_dir", type=str, default="ftlm")

    parser.add_argument("--use_node_types", action="store_true")
    parser.add_argument("--use_attributes", action="store_true")
    parser.add_argument("--use_edge_label", action="store_true")
    parser.add_argument("--use_edge_types", action="store_true")
    parser.add_argument("--use_special_tokens", action="store_true")
    parser.add_argument("--no_labels", action="store_true")

    parser.add_argument("--node_cls_label", type=str, default=None)
    parser.add_argument("--edge_cls_label", type=str, default=None)
    parser.add_argument("--cls_label", type=str, default=None)

    parser.add_argument("--min_k", type=int, default=0)
    parser.add_argument("--max_k", type=int, default=0)

    parser.add_argument("--num_epochs_llm", type=int, default=10)
    parser.add_argument("--lr_llm", type=float, default=5e-5)
    parser.add_argument("--batch_size_llm", type=int, default=32)
    parser.add_argument("--eval_batch_size_llm", type=int, default=128)

    parser.add_argument("--num_epochs_gnn", type=int, default=100)
    parser.add_argument("--lr_gnn", type=float, default=1e-3)
    parser.add_argument("--batch_size_gnn", type=int, default=32)
    parser.add_argument("--gnn_conv_model", nargs="+", default=["SAGEConv"])
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_conv_layers", type=int, default=3)
    parser.add_argument("--num_mlp_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--l_norm", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--use_embeddings", "--use_embedding", action="store_true", dest="use_embeddings")
    parser.add_argument("--use_edge_attrs", "--use_attrs", action="store_true", dest="use_edge_attrs")
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--neg_sampling_ratio", type=int, default=1)
    
    parser.add_argument("--embed_model_name", type=str, default="bert-base-uncased")

    args = parser.parse_args()
    if args.configs_file is None and args.task_id is None:
        parser.error("--task_id is required when --configs_file is not provided")
    if args.min_k > args.max_k:
        parser.error("--min_k must be <= --max_k")
    return args


def enabled_parts(args) -> Tuple[bool, bool]:
    if not args.run_llm and not args.run_gnn:
        return True, True
    return args.run_llm, args.run_gnn


def split_option(token: str) -> Tuple[str, Optional[str]]:
    if not token.startswith("--"):
        return token, None
    key = token[2:]
    if "=" in key:
        key, value = key.split("=", 1)
        key = OPTION_ALIASES.get(key, key)
        return key, value
    key = OPTION_ALIASES.get(key, key)
    return key, None


def get_task_id(tokens: List[str]) -> int:
    for index, token in enumerate(tokens):
        key, value = split_option(token)
        if key == "task_id":
            if value is not None:
                return int(value)
            return int(tokens[index + 1])
    raise ValueError(f"Missing --task_id in config: {' '.join(tokens)}")


def command_params(tokens: List[str]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key, value = split_option(token)
        if value is None and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
            index += 1
        params[key] = True if value is None else value
        index += 1
    return params


def ignored_config_flags(dataset: object) -> set:
    return DATASET_IGNORED_CONFIG_FLAGS.get(str(dataset).lower(), set())


def canonicalize_flags_for_dataset(dataset: object, flags: Dict[str, bool]) -> Dict[str, bool]:
    canonical = dict(flags)
    for flag in ignored_config_flags(dataset):
        canonical[flag] = False
    return canonical


def canonicalize_tokens_for_dataset(tokens: List[str]) -> List[str]:
    params = command_params(tokens)
    ignored_flags = ignored_config_flags(params.get("dataset", "ecore_555"))
    if not ignored_flags:
        return tokens
    return remove_args(tokens, ignored_flags)


def set_arg(tokens: List[str], key: str, value: object) -> List[str]:
    updated = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        opt_key, opt_value = split_option(token)
        if token.startswith("--") and opt_key == key:
            if opt_value is None and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                skip_next = True
            continue
        updated.append(token)
    if isinstance(value, bool):
        if value:
            updated.append(f"--{key}")
    else:
        updated.append(f"--{key}={value}")
    return updated


def remove_args(tokens: List[str], keys: Iterable[str]) -> List[str]:
    keys = set(keys)
    updated = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        key, value = split_option(token)
        if token.startswith("--") and key in keys:
            if value is None and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                skip_next = True
            continue
        updated.append(token)
    return updated


def normalize_config_line(line: str) -> List[str]:
    tokens = shlex.split(line.strip())
    for index, token in enumerate(tokens):
        key, _ = split_option(token)
        if key == "task_id":
            return tokens[index:]
    raise ValueError(f"Missing --task_id in config line: {line}")


def label_for_task(task_id: int, params: Dict[str, object]) -> str:
    if task_id in [2, 6]:
        return str(params.get("cls_label") or "label")
    if task_id in [3, 7]:
        return str(params.get("node_cls_label") or "default")
    if task_id in [5, 9]:
        return str(params.get("edge_cls_label") or "type")
    if task_id in [4, 8]:
        return "link"
    raise ValueError(f"Unsupported task_id for label inference: {task_id}")


def parse_task_args(task_id: int, tokens: List[str]):
    _, parser_factory = tasks_handler_map[task_id]
    args_without_task = remove_args(tokens, {"task_id"})
    parsed_args, unknown = parser_factory().parse_known_args(args_without_task)
    return parsed_args, unknown


def model_dir_for_tokens(tokens: List[str], models_dir: str) -> str:
    task_id = get_task_id(tokens)
    llm_task_id = GNN_TO_LLM_TASK.get(task_id, task_id)
    llm_tokens = deepcopy(tokens)
    llm_tokens = set_arg(llm_tokens, "task_id", llm_task_id)
    llm_tokens = remove_args(llm_tokens, GNN_ONLY_ARGS | EMBEDDING_ONLY_ARGS)
    llm_tokens = set_arg(llm_tokens, "models_dir", models_dir)

    parsed_args, _ = parse_task_args(llm_task_id, llm_tokens)
    config_id = get_config_hash(parsed_args)
    params = command_params(llm_tokens)
    return os.path.join(
        models_dir,
        TASK_NAMES[llm_task_id],
        str(params.get("dataset", "ecore_555")),
        label_for_task(llm_task_id, params),
        config_id,
    )


def infer_embedding_checkpoint(tokens: List[str], models_dir: str) -> List[str]:
    task_id = get_task_id(tokens)
    if task_id not in GNN_TO_LLM_TASK:
        return tokens
    params = command_params(tokens)
    if not params.get("use_embeddings") or params.get("ckpt"):
        return tokens
    models_root = str(params.get("models_dir", models_dir))
    return set_arg(tokens, "ckpt", model_dir_for_tokens(tokens, models_root))


def task_accepts_line(task_id: int, run_llm: bool, run_gnn: bool) -> bool:
    if task_id in LLM_TO_GNN_TASK:
        return run_llm
    if task_id in GNN_TO_LLM_TASK:
        return run_gnn
    return False


def ensure_link_prediction_params(tokens: List[str], neg_sampling_ratio: int) -> List[str]:
    task_id = get_task_id(tokens)
    if task_id not in [4, 8]:
        return tokens
    params = command_params(tokens)
    # if not params.get("add_negative_train_samples"):
    #     tokens = set_arg(tokens, "add_negative_train_samples", True)
    if "neg_sampling_ratio" not in params:
        tokens = set_arg(tokens, "neg_sampling_ratio", neg_sampling_ratio)
    return tokens


def read_configs_file(path: str, run_llm: bool, run_gnn: bool, models_dir: str, neg_sampling_ratio: int) -> List[Dict[str, object]]:
    items = []
    seen_config_keys = set()
    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = normalize_config_line(stripped)
            tokens = canonicalize_tokens_for_dataset(tokens)
            task_id = get_task_id(tokens)
            if not task_accepts_line(task_id, run_llm, run_gnn):
                continue
            tokens = ensure_link_prediction_params(tokens, neg_sampling_ratio)
            if task_id in GNN_TO_LLM_TASK:
                tokens = infer_embedding_checkpoint(tokens, models_dir)
            parsed_params = parsed_config_for_log(tokens)
            parsed_key = config_key(parsed_params)
            if parsed_key in seen_config_keys:
                continue
            seen_config_keys.add(parsed_key)
            items.append({
                "index": len(items),
                "source": f"{path}:{line_number}",
                "kind": "file",
                "tokens": tokens,
            })
    return items


def base_common_args(args, dataset: str, distance: int, flags: Dict[str, bool]) -> List[str]:
    command = [
        f"--dataset={dataset}",
        f"--distance={distance}",
        f"--seed={args.seed}",
        f"--test_ratio={args.test_ratio}",
        f"--min_enr={args.min_enr}",
        f"--min_edges={args.min_edges}",
        f"--language={args.language}",
        f"--limit={args.limit}",
        f"--node_topk={args.node_topk}",
        f"--results_dir={args.results_dir}",
        f"--logs_dir={args.logs_dir}",
        f"--datasets_dir={args.datasets_dir}",
        f"--models_dir={args.models_dir}",
    ]
    if args.reload:
        command.append("--reload")
    if args.remove_duplicates:
        command.append("--remove_duplicates")
    if args.include_dummies:
        command.append("--include_dummies")
    if args.use_special_tokens:
        command.append("--use_special_tokens")
    if args.no_labels:
        command.append("--no_labels")
    for flag, enabled in flags.items():
        if enabled:
            command.append(f"--{flag}")
    if args.node_cls_label:
        command.append(f"--node_cls_label={args.node_cls_label}")
    if args.edge_cls_label:
        command.append(f"--edge_cls_label={args.edge_cls_label}")
    return command


def validate_generated_labels(args):
    if args.task_id == 3 and not args.node_cls_label:
        raise ValueError("--node_cls_label is mandatory for node classification")
    if args.task_id == 5 and not args.edge_cls_label:
        raise ValueError("--edge_cls_label is mandatory for edge classification")
    if args.task_id == 2 and not args.cls_label:
        raise ValueError("--cls_label is mandatory for graph classification")


def config_flag_combinations(args, dataset: str) -> List[Dict[str, bool]]:
    ignored_flags = ignored_config_flags(dataset)
    requested = {
        flag
        for flag in CONFIG_FLAGS
        if getattr(args, flag)
    } - ignored_flags
    combinations = []
    seen = set()
    for values in itertools.product([False, True], repeat=len(CONFIG_FLAGS)):
        config = canonicalize_flags_for_dataset(dataset, dict(zip(CONFIG_FLAGS, values)))
        if all(config[flag] for flag in requested):
            key = tuple(config[flag] for flag in CONFIG_FLAGS)
            if key in seen:
                continue
            seen.add(key)
            combinations.append(config)
    return combinations


def gnn_config_combinations(args) -> List[Dict[str, bool]]:
    requested = {
        flag
        for flag in GNN_CONFIG_FLAGS
        if getattr(args, flag)
    }
    combinations = []
    for values in itertools.product([False, True], repeat=len(GNN_CONFIG_FLAGS)):
        config = dict(zip(GNN_CONFIG_FLAGS, values))
        if all(config[flag] for flag in requested):
            combinations.append(config)
    return combinations


def llm_tokens(args, dataset: str, distance: int, flags: Dict[str, bool]) -> List[str]:
    command = [
        f"--task_id={args.task_id}",
        f"--num_epochs={args.num_epochs_llm}",
        f"--lr={args.lr_llm}",
        f"--train_batch_size={args.batch_size_llm}",
        f"--eval_batch_size={args.eval_batch_size_llm}",
        f"--embed_model_name={args.embed_model_name}"
    ]
    command.extend(base_common_args(args, dataset, distance, flags))
    if args.task_id == 2:
        command.append(f"--cls_label={args.cls_label}")
    if args.task_id == 4:
        command.append(f"--neg_sampling_ratio={args.neg_sampling_ratio}")
    return command


def gnn_tokens(
    args,
    dataset: str,
    distance: int,
    flags: Dict[str, bool],
    gnn_flags: Dict[str, bool],
    model_name: str,
    paired_llm_tokens: List[str],
) -> Optional[List[str]]:
    if gnn_flags["use_edge_attrs"] and model_name not in EDGE_ATTR_GNN_MODELS:
        return None
    command = [
        f"--task_id={LLM_TO_GNN_TASK[args.task_id]}",
        f"--num_epochs={args.num_epochs_gnn}",
        f"--lr={args.lr_gnn}",
        f"--batch_size={args.batch_size_gnn}",
        f"--gnn_conv_model={model_name}",
        f"--num_conv_layers={args.num_conv_layers}",
        f"--num_mlp_layers={args.num_mlp_layers}",
        f"--hidden_dim={args.hidden_dim}",
        f"--output_dim={args.output_dim}",
        f"--dropout={args.dropout}",
        f"--aggregation={args.aggregation}",
        f"--embed_batch_size={args.embed_batch_size}",
    ]
    if LLM_TO_GNN_TASK[args.task_id] == 8:
        command.append(f"--neg_sampling_ratio={args.neg_sampling_ratio}")
    if args.num_heads is not None:
        command.append(f"--num_heads={args.num_heads}")
    elif model_name in HEADED_GNN_MODELS:
        command.append("--num_heads=4")
    if args.l_norm:
        command.append("--l_norm")
    if args.bias:
        command.append("--bias")
    if gnn_flags["use_edge_attrs"]:
        command.append("--use_edge_attrs")
    if gnn_flags["use_embeddings"]:
        command.append("--use_embeddings")
        command.append(f"--ckpt={model_dir_for_tokens(paired_llm_tokens, args.models_dir)}")
    command.extend(base_common_args(args, dataset, distance, flags))
    if args.task_id == 2:
        command.append(f"--cls_label={args.cls_label}")
    return command


def generated_configs(args, run_llm: bool, run_gnn: bool) -> List[Dict[str, object]]:
    validate_generated_labels(args)
    items = []
    base_index = 0
    for dataset in args.dataset:
        for distance, flags in itertools.product(
            range(args.min_k, args.max_k + 1),
            config_flag_combinations(args, dataset),
        ):
            paired_llm = llm_tokens(args, dataset, distance, flags)
            if run_llm:
                items.append({
                    "index": base_index,
                    "source": "generated",
                    "kind": "llm",
                    "tokens": paired_llm,
                })
            if run_gnn:
                for gnn_flags, model_name in itertools.product(
                    gnn_config_combinations(args),
                    args.gnn_conv_model,
                ):
                    paired_gnn = gnn_tokens(
                        args,
                        dataset,
                        distance,
                        flags,
                        gnn_flags,
                        model_name,
                        paired_llm,
                    )
                    if paired_gnn is not None:
                        items.append({
                            "index": base_index,
                            "source": "generated",
                            "kind": "gnn",
                            "tokens": paired_gnn,
                        })
            base_index += 1
    return items


def slice_items(items: List[Dict[str, object]], start: int, end: int) -> List[Dict[str, object]]:
    if end == -1:
        end = len(items)
    return items[start:end]


def is_memory_error(stderr: str, stdout: str) -> bool:
    text = f"{stderr}\n{stdout}".lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "cublas_status_alloc_failed" in text
        or "cuda out of memory" in text
    )


def batch_arg_for_task(task_id: int) -> str:
    return "train_batch_size" if task_id in LLM_TO_GNN_TASK else "batch_size"


def current_batch_size(tokens: List[str], key: str) -> Optional[int]:
    params = command_params(tokens)
    if key not in params:
        return None
    try:
        return int(params[key])
    except (TypeError, ValueError):
        return None


def retry_with_smaller_batch(tokens: List[str]) -> Optional[List[str]]:
    task_id = get_task_id(tokens)
    batch_key = batch_arg_for_task(task_id)
    batch_size = current_batch_size(tokens, batch_key)
    if batch_size is None or batch_size <= 8:
        return None
    next_batch_size = max(8, batch_size // 2)
    if next_batch_size == batch_size:
        return None
    return set_arg(tokens, batch_key, next_batch_size)


def parsed_config_for_log(tokens: List[str]) -> Dict[str, object]:
    task_id = get_task_id(tokens)
    parsed_args, unknown = parse_task_args(task_id, tokens)
    payload = vars(parsed_args)
    payload["task_id"] = task_id
    if unknown:
        payload["unknown_args"] = unknown
    return payload


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def json_dumps(value) -> str:
    return json.dumps(value, default=json_default, sort_keys=True)


def config_key(config_params: Dict[str, object]) -> str:
    return md5_hash(json_dumps(config_params))


def read_status_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def completed_config_keys(path: str) -> set:
    return {
        row.get("config_key")
        for row in read_status_rows(path)
        if row.get("execution_status") == "success" and row.get("config_key")
    }


def flatten_results(value, prefix="result") -> Dict[str, object]:
    flattened = {}
    if isinstance(value, list):
        if not value:
            return flattened
        dict_items = [item for item in value if isinstance(item, dict)]
        test_items = [item for item in dict_items if item.get("phase") == "test"]
        best = max(
            test_items or dict_items,
            key=lambda item: item.get("balanced_accuracy", float("-inf")),
            default=None,
        )
        if best is not None:
            flattened.update(flatten_results(best, prefix))
        flattened[f"{prefix}_num_records"] = len(value)
        return flattened
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).replace("/", "_")
            if isinstance(item, dict):
                flattened.update(flatten_results(item, f"{prefix}_{normalized_key}"))
            elif isinstance(item, list):
                flattened[f"{prefix}_{normalized_key}"] = json_dumps(item)
            else:
                flattened[f"{prefix}_{normalized_key}"] = item
    return flattened


def experiment_dir_for_config(config_params: Dict[str, object]) -> str:
    task_id = int(config_params["task_id"])
    task_name = TASK_NAMES[task_id]
    if task_id == 6 and config_params.get("include_dummies"):
        task_name = "GNN_dummy_graph_cls"
    return os.path.join(
        str(config_params.get("results_dir", "results")),
        task_name,
        str(config_params.get("dataset", config_params.get("dataset_name", "default"))),
        label_for_task(task_id, config_params),
        get_config_str(argparse.Namespace(**config_params)) or "default",
    )


def load_final_results(config_params: Dict[str, object]) -> Tuple[object, Dict[str, object]]:
    results_path = os.path.join(experiment_dir_for_config(config_params), RESULTS_FILE_NAME)
    if not os.path.exists(results_path):
        return None, {}
    with open(results_path, "r") as f:
        results = json.load(f)
    return results, flatten_results(results)


def flatten_config_params(config_params: Dict[str, object]) -> Dict[str, object]:
    flattened = {}
    for key, value in config_params.items():
        if isinstance(value, (dict, list)):
            flattened[f"config_{key}"] = json_dumps(value)
        else:
            flattened[f"config_{key}"] = value
    return flattened


def append_status(path: str, record: Dict[str, object]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = read_status_rows(path)
    serialized = {
        key: json_dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in record.items()
    }
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    for key in serialized.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    rows.append(serialized)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_runtime_config(tokens: List[str]) -> Optional[str]:
    task_id = get_task_id(tokens)
    params = command_params(tokens)
    if task_id in GNN_TO_LLM_TASK and params.get("use_edge_attrs"):
        model_name = str(params.get("gnn_conv_model", "SAGEConv"))
        if model_name not in EDGE_ATTR_GNN_MODELS:
            return (
                f"--use_edge_attrs requires an edge-attribute-aware GNN encoder; "
                f"got {model_name}. Supported: {sorted(EDGE_ATTR_GNN_MODELS)}"
            )
    return None


def run_streaming(command: List[str], env: Dict[str, str]) -> Dict[str, object]:
    process = subprocess.Popen(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_chunks = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_chunks.append(line)
    returncode = process.wait()
    output = "".join(output_chunks)
    return {
        "returncode": returncode,
        "stdout": output,
        "stderr": "",
    }


def execute_item(item: Dict[str, object], args, env: Dict[str, str]) -> Dict[str, object]:
    tokens = list(item["tokens"])
    attempts = []
    validation_error = validate_runtime_config(tokens)
    if validation_error:
        return {
            "status": "failure",
            "error": validation_error,
            "attempts": attempts,
            "final_tokens": tokens,
        }

    while True:
        command = [sys.executable, "-m", "glam4cm.run"] + tokens
        print("Running:", " ".join(shlex.quote(part) for part in command))
        if args.dry_run:
            return {
                "status": "success",
                "dry_run": True,
                "attempts": [{"command": command, "returncode": 0}],
                "final_tokens": tokens,
            }

        result = run_streaming(command, env)
        attempts.append({
            "command": command,
            "returncode": result["returncode"],
            "stdout_tail": result["stdout"][-4000:],
            "stderr_tail": result["stderr"][-4000:],
        })
        if result["returncode"] == 0:
            return {
                "status": "success",
                "attempts": attempts,
                "final_tokens": tokens,
            }

        next_tokens = retry_with_smaller_batch(tokens) \
            if is_memory_error(result["stderr"], result["stdout"]) else None
        if next_tokens is None:
            return {
                "status": "failure",
                "attempts": attempts,
                "final_tokens": tokens,
                "error": result["stderr"][-4000:] or result["stdout"][-4000:],
            }
        tokens = next_tokens
        print("Memory failure detected; retrying with smaller batch size.")


def execute_configs(items: List[Dict[str, object]], args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.cuda_devices)
    print("CUDA_VISIBLE_DEVICES=", env["CUDA_VISIBLE_DEVICES"])
    print(f"Total configurations to run: {len(items)}")
    completed_keys = completed_config_keys(args.status_file) if not args.rerun else set()

    for item in tqdm(items, desc="Running configurations"):
        tokens = list(item["tokens"])
        task_id = get_task_id(tokens)
        initial_config_params = parsed_config_for_log(tokens)
        initial_config_key = config_key(initial_config_params)
        if initial_config_key in completed_keys:
            print(f"Skipping previously successful configuration: {' '.join(tokens)}")
            continue

        started_at = datetime.now(timezone.utc).isoformat()
        result = execute_item(item, args, env)
        finished_at = datetime.now(timezone.utc).isoformat()
        final_config_params = parsed_config_for_log(result["final_tokens"])
        final_results, flattened_results = load_final_results(final_config_params) \
            if result["status"] == "success" and not args.dry_run else (None, {})
        record = {
            "task_id": task_id,
            "task_name": tasks.get(task_id),
            "timestamp": finished_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "source": item["source"],
            "kind": item["kind"],
            "config_index": item["index"],
            "config_key": initial_config_key,
            "final_config_key": config_key(final_config_params),
            "command": " ".join(tokens),
            "final_command": " ".join(result["final_tokens"]),
            "config_params": final_config_params,
            "execution_status": result["status"],
            "attempts": result["attempts"],
            "final_results": final_results,
        }
        record.update(flatten_config_params(final_config_params))
        record.update(flattened_results)
        if "error" in result:
            record["error"] = result["error"]
        append_status(args.status_file, record)
        if result["status"] == "success":
            completed_keys.add(initial_config_key)
        if result["status"] == "failure":
            print(f"Configuration failed: {' '.join(tokens)}")


def main():
    args = parse_args()
    run_llm, run_gnn = enabled_parts(args)
    if args.configs_file:
        items = read_configs_file(
            args.configs_file,
            run_llm,
            run_gnn,
            args.models_dir,
            args.neg_sampling_ratio,
        )
    else:
        try:
            items = generated_configs(args, run_llm, run_gnn)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    items = slice_items(items, args.start, args.end)

    for index, item in enumerate(items):
        item["run_index"] = index

    execute_configs(items, args)


if __name__ == "__main__":
    main()
