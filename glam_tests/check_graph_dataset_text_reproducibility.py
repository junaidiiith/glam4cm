#!/usr/bin/env python3
"""Check graph_dataset text/label generation across repeated task runs."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from glam4cm.data_loading.graph_dataset import GraphEdgeDataset, GraphNodeDataset
from glam4cm.data_loading.models_dataset import get_models_dataset
from glam4cm.settings import EDGE_CLS_TASK, GRAPH_CLS_TASK, LINK_PRED_TASK, NODE_CLS_TASK


DEFAULT_NODE_LABELS = {
    "ecore_555": "abstract",
    "mar-ecore-github": "abstract",
    "modelset": "abstract",
    "eamodelset": "type",
    "ontouml": "stereotype",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate graph_dataset text/label pairs for each task and compare "
            "them across repeated runs."
        )
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="ecore_555", choices=sorted(DEFAULT_NODE_LABELS))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--distance", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--node-label", default=None)
    parser.add_argument("--edge-label", default="type")
    parser.add_argument("--min-enr", type=float, default=-1.0)
    parser.add_argument("--min-edges", type=int, default=-1)
    parser.add_argument("--language", default="en")
    parser.add_argument("--remove-duplicates", action="store_true")
    parser.add_argument("--include-dummies", action="store_true")
    parser.add_argument("--use-attributes", action="store_true")
    parser.add_argument("--use-edge-label", action="store_true")
    parser.add_argument("--use-edge-types", action="store_true")
    parser.add_argument("--use-node-types", action="store_true")
    parser.add_argument("--use-special-tokens", action="store_true")
    parser.add_argument(
        "--reuse-graph-cache",
        action="store_true",
        help="Reuse task graph cache files after the first run instead of regenerating them.",
    )
    parser.add_argument(
        "--graph-save-dir",
        default=None,
        help="Optional graph cache directory. A temporary directory is used by default.",
    )
    parser.add_argument(
        "--snapshot-json",
        default=None,
        help="Write the baseline task snapshots to this JSON file.",
    )
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    return args


def to_python(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [to_python(item) for item in value]
    if isinstance(value, list):
        return [to_python(item) for item in value]
    return value


def text_label_pairs(texts: Sequence[str], labels: Sequence[Any]) -> List[Dict[str, Any]]:
    if len(texts) != len(labels):
        raise AssertionError(f"Text/label length mismatch: {len(texts)} != {len(labels)}")
    return [
        {
            "text": text,
            "label": to_python(label),
        }
        for text, label in zip(texts, labels)
    ]


def graph_common_params(args: argparse.Namespace, save_dir: str) -> Dict[str, Any]:
    return {
        "distance": args.distance,
        "test_ratio": args.test_ratio,
        "reload": not args.reuse_graph_cache,
        "use_attributes": args.use_attributes,
        "use_edge_label": args.use_edge_label,
        "use_edge_types": args.use_edge_types,
        "use_node_types": args.use_node_types,
        "use_special_tokens": args.use_special_tokens,
        "limit": args.limit,
        "save_dir": save_dir,
        "seed": args.seed,
        "node_cls_label": args.node_label or DEFAULT_NODE_LABELS[args.dataset],
        "edge_cls_label": args.edge_label,
    }


def load_models_dataset(args: argparse.Namespace):
    return get_models_dataset(
        args.dataset,
        include_dummies=args.include_dummies,
        min_enr=args.min_enr,
        min_edges=args.min_edges,
        remove_duplicates=args.remove_duplicates,
        language=args.language,
    )


def snapshot_node_classification(args: argparse.Namespace, save_dir: str) -> Dict[str, Any]:
    graph_dataset = GraphNodeDataset(
        load_models_dataset(args),
        task_type=NODE_CLS_TASK,
        **graph_common_params(args, save_dir),
    )
    texts = graph_dataset.get_node_classification_texts(
        distance=args.distance,
        label=args.node_label or DEFAULT_NODE_LABELS[args.dataset],
    )
    return {
        "train_nodes": text_label_pairs(texts["train_nodes"], texts["train_node_classes"]),
        "test_nodes": text_label_pairs(texts["test_nodes"], texts["test_node_classes"]),
    }


def snapshot_edge_classification(args: argparse.Namespace, save_dir: str) -> Dict[str, Any]:
    graph_dataset = GraphEdgeDataset(
        load_models_dataset(args),
        task_type=EDGE_CLS_TASK,
        **graph_common_params(args, save_dir),
    )
    texts = graph_dataset.get_edges_texts(label=args.edge_label)
    return {
        "train_edges": text_label_pairs(texts["train_pos_edges"], texts["train_edge_classes"]),
        "test_edges": text_label_pairs(texts["test_pos_edges"], texts["test_edge_classes"]),
    }


def link_prediction_pairs(texts: Dict[str, Sequence[str]], split: str) -> List[Dict[str, Any]]:
    return (
        text_label_pairs(texts[f"{split}_pos_edges"], [1] * len(texts[f"{split}_pos_edges"]))
        + text_label_pairs(texts[f"{split}_neg_edges"], [0] * len(texts[f"{split}_neg_edges"]))
    )


def snapshot_link_prediction(args: argparse.Namespace, save_dir: str) -> Dict[str, Any]:
    graph_dataset = GraphEdgeDataset(
        load_models_dataset(args),
        task_type=LINK_PRED_TASK,
        add_negative_train_samples=True,
        **graph_common_params(args, save_dir),
    )
    texts = graph_dataset.get_edges_texts()
    return {
        "train_edges": link_prediction_pairs(texts, "train"),
        "test_edges": link_prediction_pairs(texts, "test"),
    }


def snapshot_graph_classification(args: argparse.Namespace, save_dir: str) -> Dict[str, Any]:
    graph_dataset = GraphNodeDataset(
        load_models_dataset(args),
        task_type=GRAPH_CLS_TASK,
        **graph_common_params(args, save_dir),
    )
    graph_text_label = graph_dataset.metadata.graph_label
    graph_class_label = graph_dataset.metadata.graph_cls
    if not graph_text_label or not graph_class_label:
        return {
            "skipped": (
                f"{args.dataset} metadata does not define both graph text and "
                "graph classification labels."
            )
        }

    graph_texts = [getattr(graph, graph_text_label) for graph in graph_dataset.graphs]
    graph_labels = [
        getattr(graph.data, f"graph_{graph_class_label}")[0]
        for graph in graph_dataset.graphs
    ]
    return {"graphs": text_label_pairs(graph_texts, graph_labels)}


def collect_snapshots(args: argparse.Namespace, save_dir: str) -> Dict[str, Dict[str, Any]]:
    return {
        NODE_CLS_TASK: snapshot_node_classification(args, save_dir),
        EDGE_CLS_TASK: snapshot_edge_classification(args, save_dir),
        LINK_PRED_TASK: snapshot_link_prediction(args, save_dir),
        GRAPH_CLS_TASK: snapshot_graph_classification(args, save_dir),
    }


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def section_sizes(snapshot: Dict[str, Any]) -> Dict[str, int]:
    return {
        name: len(records)
        for name, records in snapshot.items()
        if isinstance(records, list)
    }


def first_difference(expected: Any, actual: Any, path: Iterable[str] = ()) -> str:
    here = ".".join(path) or "snapshot"
    if type(expected) is not type(actual):
        return f"{here}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return f"{here}: keys {sorted(expected)} != {sorted(actual)}"
        for key in expected:
            difference = first_difference(expected[key], actual[key], (*path, str(key)))
            if difference:
                return difference
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{here}: length {len(expected)} != {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = first_difference(
                expected_item,
                actual_item,
                (*path, f"[{index}]"),
            )
            if difference:
                return difference
        return ""
    if expected != actual:
        return f"{here}: {expected!r} != {actual!r}"
    return ""


def write_snapshot(path: str, snapshot: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_check(args: argparse.Namespace, save_dir: str) -> None:
    baseline = collect_snapshots(args, save_dir)
    baseline_fingerprints = {
        task: fingerprint(snapshot)
        for task, snapshot in baseline.items()
    }
    print("Run 1 baseline:")
    for task, snapshot in baseline.items():
        print(f"  {task}: {baseline_fingerprints[task]} {section_sizes(snapshot)}")

    for run_number in range(2, args.runs + 1):
        current = collect_snapshots(args, save_dir)
        print(f"Run {run_number}:")
        for task, snapshot in current.items():
            current_fingerprint = fingerprint(snapshot)
            print(f"  {task}: {current_fingerprint} {section_sizes(snapshot)}")
            if snapshot != baseline[task]:
                difference = first_difference(baseline[task], snapshot, (task,))
                raise AssertionError(
                    f"{task} text/label generation changed on run {run_number}: "
                    f"{difference}"
                )

    if args.snapshot_json:
        write_snapshot(args.snapshot_json, baseline)
        print(f"Wrote baseline snapshot to {args.snapshot_json}")
    print(f"All graph_dataset text/label snapshots matched across {args.runs} runs.")


def main() -> None:
    args = parse_args()
    if args.graph_save_dir:
        Path(args.graph_save_dir).mkdir(parents=True, exist_ok=True)
        run_check(args, args.graph_save_dir)
        return

    with tempfile.TemporaryDirectory(prefix="glam4cm-graph-text-repro-") as save_dir:
        run_check(args, save_dir)


if __name__ == "__main__":
    main()
