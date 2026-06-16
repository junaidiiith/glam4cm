import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from glam4cm.utils import md5_hash


def _shape(value):
    if value is None:
        return None
    if hasattr(value, "shape"):
        return list(value.shape)
    return None


def _tensor_stats(value) -> Dict[str, object]:
    if value is None:
        return {"shape": None}
    tensor = torch.as_tensor(value)
    tensor_cpu = tensor.detach().cpu().contiguous()
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "hash": hashlib.md5(tensor_cpu.numpy().tobytes()).hexdigest(),
    }
    if tensor.numel() > 0 and tensor.is_floating_point():
        stats.update({
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        })
    return stats


def _hash_strings(values: Iterable[str]) -> str:
    return md5_hash("\n".join(values))


def _sample_mapping(mapping: Dict[object, str], limit: int = 5) -> List[Dict[str, object]]:
    samples = []
    for key, value in list(mapping.items())[:limit]:
        samples.append({
            "key": str(key),
            "text": value,
            "text_len": len(value),
        })
    return samples


def _edge_label_counts(graph_dataset) -> Dict[str, object]:
    label = graph_dataset.metadata.edge_cls
    train_counter = Counter()
    test_counter = Counter()
    for graph in graph_dataset.graphs:
        values = getattr(graph.data, f"edge_{label}", None)
        if values is None:
            continue
        train_counter.update(values[graph.data.train_edge_mask].tolist())
        test_counter.update(values[graph.data.test_edge_mask].tolist())
    return {
        "label": label,
        "train": {str(key): int(value) for key, value in sorted(train_counter.items())},
        "test": {str(key): int(value) for key, value in sorted(test_counter.items())},
    }


def build_link_prediction_diagnostics(graph_dataset, args=None) -> Dict[str, object]:
    graph_summaries = []
    total_train_pos = 0
    total_train_neg = 0
    total_test_pos = 0
    total_test_neg = 0
    node_text_hashes = []
    edge_text_hashes = []

    for index, graph in enumerate(graph_dataset.graphs):
        data = graph.data
        train_pos = data.train_pos_edge_label_index.shape[1]
        train_neg = data.train_neg_edge_label_index.shape[1]
        test_pos = data.test_pos_edge_label_index.shape[1]
        test_neg = data.test_neg_edge_label_index.shape[1]
        total_train_pos += train_pos
        total_train_neg += train_neg
        total_test_pos += test_pos
        total_test_neg += test_neg

        node_text_values = list(graph.node_texts.values())
        edge_text_values = list(graph.edge_texts.values())
        node_text_hash = _hash_strings(node_text_values)
        edge_text_hash = _hash_strings(edge_text_values)
        node_text_hashes.append(node_text_hash)
        edge_text_hashes.append(edge_text_hash)

        if index < 5:
            graph_summaries.append({
                "index": index,
                "name": graph.name,
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.num_edges),
                "train_pos_edges": int(train_pos),
                "train_neg_edges": int(train_neg),
                "test_pos_edges": int(test_pos),
                "test_neg_edges": int(test_neg),
                "edge_index_shape": _shape(data.edge_index),
                "overall_edge_index_shape": _shape(data.overall_edge_index),
                "x": _tensor_stats(data.x),
                "edge_attr": _tensor_stats(data.edge_attr),
                "node_text_hash": node_text_hash,
                "edge_text_hash": edge_text_hash,
                "node_text_samples": _sample_mapping(graph.node_texts),
                "edge_text_samples": _sample_mapping(graph.edge_texts),
            })

    config = dict(graph_dataset.config)
    if args is not None:
        config["lp_message_passing_edges"] = getattr(args, "lp_message_passing_edges", None)

    return {
        "config": config,
        "config_hash": graph_dataset.config_hash,
        "string_gen_params_hash": graph_dataset.get_string_gen_params_hash(),
        "num_graphs": len(graph_dataset.graphs),
        "totals": {
            "train_pos_edges": int(total_train_pos),
            "train_neg_edges": int(total_train_neg),
            "test_pos_edges": int(total_test_pos),
            "test_neg_edges": int(total_test_neg),
            "unique_node_text_hashes": len(set(node_text_hashes)),
            "unique_edge_text_hashes": len(set(edge_text_hashes)),
            "combined_node_text_hash": _hash_strings(node_text_hashes),
            "combined_edge_text_hash": _hash_strings(edge_text_hashes),
        },
        "edge_label_counts": _edge_label_counts(graph_dataset),
        "graphs": graph_summaries,
        "notes": [
            "For non-embedding GNN runs, x and edge_attr are random vectors plus optional type one-hot features.",
            "In the current GraphDataset post-processing, use_edge_types is not appended to edge_attr for LINK_PRED_TASK.",
            "Distance k changes generated node/edge text. It only affects GNN tensors when use_embeddings is enabled or when text-derived features are otherwise injected.",
        ],
    }


def write_link_prediction_diagnostics(graph_dataset, args=None, output_path=None) -> Dict[str, object]:
    diagnostics = build_link_prediction_diagnostics(graph_dataset, args=args)
    text = json.dumps(diagnostics, indent=2, sort_keys=True)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote GNN link prediction diagnostics to {path}")
    else:
        print(text)
    return diagnostics
