#!/usr/bin/env python3
"""Build link prediction F1 tables from run_configurations_status.csv."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BERT_LINK_TASK_ID = 4
GNN_LINK_TASK_ID = 8
DEFAULT_DISTANCES = [0, 1, 2, 3]

FLAG_COLUMNS = [
    ("config_use_attributes", "N_A"),
    ("config_use_node_types", "N_T"),
    ("config_use_edge_label", "E_L"),
    ("config_use_edge_types", "E_T"),
]

MODEL_NAMES = {
    BERT_LINK_TASK_ID: "BERT",
    GNN_LINK_TASK_ID: "GNN",
}

MODEL_ORDER = {
    "BERT": 0,
    "GNN": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Table-9-style link prediction F1 table from "
            "logs/run_configurations_status.csv."
        )
    )
    parser.add_argument(
        "--input",
        default="logs/run_configurations_status.csv",
        help="Path to run_configurations_status.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to stdout unless --format all is used.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "csv", "all"],
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=DEFAULT_DISTANCES,
        help="k values to include as columns.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Optional dataset filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimal places for F1 scores.",
    )
    parser.add_argument(
        "--bold_best",
        action="store_true",
        help="Bold the best score within each dataset/model/k column.",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int(value: object) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def parse_float(value: object) -> Optional[float]:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def dataset_label(dataset: str) -> str:
    labels = {
        "ecore_555": "Ecore-555",
        "eamodelset": "EAModelSet",
        "modelset": "ModelSet",
    }
    return labels.get(dataset, dataset)


def score_for_row(row: Dict[str, str], task_id: int) -> Optional[float]:
    if task_id == BERT_LINK_TASK_ID:
        return parse_float(row.get("result_eval_f1_macro"))
    if task_id == GNN_LINK_TASK_ID:
        return parse_float(row.get("result_f1_macro"))
    return None


def config_key(row: Dict[str, str]) -> Tuple[bool, bool, bool, bool]:
    return tuple(parse_bool(row.get(column)) for column, _ in FLAG_COLUMNS)


def config_label(flags: Tuple[bool, bool, bool, bool]) -> str:
    parts = ["N_L"]
    for enabled, (_, label) in zip(flags, FLAG_COLUMNS):
        if enabled:
            parts.append(label)
    return "+".join(parts)


def config_sort_key(flags: Tuple[bool, bool, bool, bool]) -> Tuple[int, Tuple[int, ...]]:
    enabled_positions = tuple(index for index, enabled in enumerate(flags) if enabled)
    return (sum(flags), enabled_positions)


def load_best_scores(
    path: Path,
    datasets: Optional[Iterable[str]],
) -> Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float]:
    dataset_filter = set(datasets) if datasets else None
    best_scores: Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("execution_status") != "success":
                continue

            task_id = parse_int(row.get("task_id") or row.get("config_task_id"))
            if task_id not in MODEL_NAMES:
                continue

            dataset = (row.get("config_dataset") or row.get("dataset") or "").strip()
            if not dataset or (dataset_filter and dataset not in dataset_filter):
                continue

            distance = parse_int(row.get("config_distance"))
            if distance is None:
                continue

            score = score_for_row(row, task_id)
            if score is None:
                continue

            key = (dataset, MODEL_NAMES[task_id], config_key(row), distance)
            best_scores[key] = max(score, best_scores.get(key, float("-inf")))

    return best_scores


def table_rows(
    best_scores: Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float],
) -> List[Tuple[str, str, Tuple[bool, bool, bool, bool]]]:
    rows = {
        (dataset, model, flags)
        for dataset, model, flags, _ in best_scores
    }
    return sorted(
        rows,
        key=lambda item: (
            dataset_label(item[0]).lower(),
            MODEL_ORDER.get(item[1], 99),
            config_sort_key(item[2]),
        ),
    )


def best_by_column(
    best_scores: Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float],
) -> Dict[Tuple[str, str, int], float]:
    grouped: Dict[Tuple[str, str, int], float] = {}
    for dataset, model, flags, distance in best_scores:
        key = (dataset, model, distance)
        value = best_scores[(dataset, model, flags, distance)]
        grouped[key] = max(grouped.get(key, float("-inf")), value)
    return grouped


def format_score(
    value: Optional[float],
    decimals: int,
    bold: bool = False,
    latex: bool = False,
) -> str:
    if value is None:
        return "-"
    rendered = f"{value:.{decimals}f}"
    if not bold:
        return rendered
    return f"\\textbf{{{rendered}}}" if latex else f"**{rendered}**"


def build_records(
    best_scores: Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float],
    distances: List[int],
    decimals: int,
    bold_best: bool,
    latex: bool = False,
) -> List[List[str]]:
    column_bests = best_by_column(best_scores) if bold_best else {}
    records: List[List[str]] = []
    for dataset, model, flags in table_rows(best_scores):
        record = [dataset_label(dataset), model, config_label(flags)]
        for distance in distances:
            value = best_scores.get((dataset, model, flags, distance))
            bold = (
                value is not None
                and bold_best
                and value == column_bests.get((dataset, model, distance))
            )
            record.append(format_score(value, decimals, bold=bold, latex=latex))
        records.append(record)
    return records


def build_markdown(
    best_scores: Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float],
    distances: List[int],
    decimals: int,
    bold_best: bool,
) -> str:
    headers = ["Dataset", "Model", "Config", *[f"k = {distance}" for distance in distances]]
    records = build_records(best_scores, distances, decimals, bold_best)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|" + "|".join(["---:"] * len(distances)) + "|",
    ]
    lines.extend("| " + " | ".join(record) + " |" for record in records)
    lines.extend(
        [
            "",
            "Notes: N_L = node label; N_A = use_node_attributes; "
            "N_T = use_node_types; EL = use_edge_label; ET = use_edge_types.",
            "Duplicate cells keep the run with the higher F1 score.",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def latex_config(value: str) -> str:
    parts = value.split("+")
    formatted = []
    for part in parts:
        if "_" in part:
            head, tail = part.split("_", 1)
            formatted.append(f"{head}_{{{tail}}}")
        else:
            formatted.append(part)
    return "$" + "+".join(formatted) + "$"


def build_latex(
    best_scores: Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float],
    distances: List[int],
    decimals: int,
    bold_best: bool,
) -> str:
    headers = ["Dataset", "Model", "Config", *[f"$k = {distance}$" for distance in distances]]
    records = build_records(best_scores, distances, decimals, bold_best, latex=True)
    column_spec = "lll" + "r" * len(distances)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Link Prediction F1-Scores Results}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    previous_dataset = None
    for record in records:
        dataset, model, config, *scores = record
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append("\\midrule")
        previous_dataset = dataset
        rendered = [latex_escape(dataset), model, latex_config(config), *scores]
        lines.append(" & ".join(rendered) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[0.4em]",
            "\\footnotesize Notes: $N_L$ = node label; $N_A$ = use\\_node\\_attributes; "
            "$N_T$ = use\\_node\\_types; EL = use\\_edge\\_label; ET = use\\_edge\\_types. "
            "Duplicate cells keep the run with the higher F1 score.",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_csv(
    best_scores: Dict[Tuple[str, str, Tuple[bool, bool, bool, bool], int], float],
    distances: List[int],
    decimals: int,
) -> str:
    rows = build_records(best_scores, distances, decimals, bold_best=False)
    headers = ["Dataset", "Model", "Config", *[f"k={distance}" for distance in distances]]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def write_output(path: Optional[str], text: str) -> None:
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main() -> int:
    args = parse_args()
    best_scores = load_best_scores(Path(args.input), args.dataset)
    if not best_scores:
        raise SystemExit("No successful BERT/GNN link prediction rows found.")

    if args.format == "markdown":
        write_output(args.output, build_markdown(best_scores, args.distances, args.decimals, args.bold_best))
        return 0
    if args.format == "latex":
        write_output(args.output, build_latex(best_scores, args.distances, args.decimals, args.bold_best))
        return 0
    if args.format == "csv":
        write_output(args.output, build_csv(best_scores, args.distances, args.decimals))
        return 0

    output_base = Path(args.output or "link_prediction_f1_table")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".md").write_text(
        build_markdown(best_scores, args.distances, args.decimals, args.bold_best),
        encoding="utf-8",
    )
    output_base.with_suffix(".tex").write_text(
        build_latex(best_scores, args.distances, args.decimals, args.bold_best),
        encoding="utf-8",
    )
    output_base.with_suffix(".csv").write_text(
        build_csv(best_scores, args.distances, args.decimals),
        encoding="utf-8",
    )
    print(f"wrote {output_base.with_suffix('.md')}")
    print(f"wrote {output_base.with_suffix('.tex')}")
    print(f"wrote {output_base.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
