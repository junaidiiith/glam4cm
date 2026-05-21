import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path("results_graph_cls.csv")
OUTPUT_CSV = Path("runtime_trend_summary.csv")
OUTPUT_MD = Path("runtime_trend_summary.md")
OUTPUT_SVG = Path("runtime_trend_heatmap.svg")


def load_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            runtime = row.get("train/train_runtime", "").strip()
            dataset = row.get("dataset", "").strip()
            task_type = row.get("task_type", "").strip()
            distance = row.get("distance", "").strip()
            if not runtime or not dataset or not task_type or distance == "":
                continue
            try:
                runtime_value = float(runtime)
                distance_value = int(float(distance))
            except ValueError:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "distance": distance_value,
                    "runtime": runtime_value,
                }
            )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["task_type"], row["distance"])
        grouped[key].append(row["runtime"])

    summary = {}
    for key, values in grouped.items():
        summary[key] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def write_summary_csv(summary):
    keys = sorted(summary)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "task_type",
                "distance",
                "runs",
                "mean_runtime_sec",
                "median_runtime_sec",
                "min_runtime_sec",
                "max_runtime_sec",
            ]
        )
        for dataset, task_type, distance in keys:
            stats = summary[(dataset, task_type, distance)]
            writer.writerow(
                [
                    dataset,
                    task_type,
                    distance,
                    stats["count"],
                    round(stats["mean"], 2),
                    round(stats["median"], 2),
                    round(stats["min"], 2),
                    round(stats["max"], 2),
                ]
            )


def build_markdown(summary):
    pair_keys = sorted({(dataset, task_type) for dataset, task_type, _ in summary})
    distances = sorted({distance for _, _, distance in summary})
    lines = [
        "# Runtime Trend Summary",
        "",
        "Mean training runtime in seconds by dataset, task type, and distance.",
        "",
        "| Dataset | Task Type | D0 | D1 | D2 | D3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, task_type in pair_keys:
        cells = [dataset, task_type]
        for distance in distances:
            stats = summary.get((dataset, task_type, distance))
            cells.append(f"{stats['mean']:.1f}" if stats else "-")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "Run counts by cell are available in `runtime_trend_summary.csv`.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def color_for_value(value, min_value, max_value):
    if max_value <= min_value:
        t = 0.5
    else:
        t = (value - min_value) / (max_value - min_value)
    start = (240, 249, 232)
    end = (33, 113, 181)
    rgb = tuple(round(start[i] + t * (end[i] - start[i])) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def text_color_for_value(value, min_value, max_value):
    if max_value <= min_value:
        t = 0.5
    else:
        t = (value - min_value) / (max_value - min_value)
    return "white" if t > 0.58 else "#0f172a"


def write_svg(summary):
    pair_keys = sorted({(dataset, task_type) for dataset, task_type, _ in summary})
    distances = sorted({distance for _, _, distance in summary})
    values = [stats["mean"] for stats in summary.values()]
    min_value = min(values)
    max_value = max(values)

    left_pad = 240
    top_pad = 70
    cell_w = 110
    cell_h = 38
    width = left_pad + cell_w * len(distances) + 40
    height = top_pad + cell_h * len(pair_keys) + 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd" />',
        '<text x="24" y="34" font-family="Arial, Helvetica, sans-serif" font-size="22" fill="#0f172a">Runtime Trend Heatmap</text>',
        '<text x="24" y="54" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#475569">Mean training runtime in seconds by dataset, task type, and distance</text>',
    ]

    for col, distance in enumerate(distances):
        x = left_pad + col * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top_pad - 16}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#334155">distance={distance}</text>'
        )

    for row, (dataset, task_type) in enumerate(pair_keys):
        y = top_pad + row * cell_h
        label = f"{dataset} / {task_type}"
        parts.append(
            f'<text x="{left_pad - 12}" y="{y + 24}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#0f172a">{label}</text>'
        )
        for col, distance in enumerate(distances):
            x = left_pad + col * cell_w
            stats = summary.get((dataset, task_type, distance))
            if stats:
                value = stats["mean"]
                fill = color_for_value(value, min_value, max_value)
                text_fill = text_color_for_value(value, min_value, max_value)
                text = f"{value:.0f}s"
            else:
                fill = "#e2e8f0"
                text_fill = "#64748b"
                text = "-"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="6" fill="{fill}" stroke="#ffffff" />'
            )
            parts.append(
                f'<text x="{x + (cell_w - 4) / 2}" y="{y + 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{text_fill}">{text}</text>'
            )

    legend_x = 24
    legend_y = height - 28
    legend_w = 240
    steps = 24
    for i in range(steps):
        x = legend_x + i * (legend_w / steps)
        value = min_value + (i / max(steps - 1, 1)) * (max_value - min_value)
        parts.append(
            f'<rect x="{x:.2f}" y="{legend_y}" width="{(legend_w / steps) + 1:.2f}" height="12" fill="{color_for_value(value, min_value, max_value)}" stroke="none" />'
        )
    parts.append(
        f'<text x="{legend_x}" y="{legend_y - 6}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#475569">Low: {min_value:.1f}s</text>'
    )
    parts.append(
        f'<text x="{legend_x + legend_w}" y="{legend_y - 6}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#475569">High: {max_value:.1f}s</text>'
    )
    parts.append("</svg>")
    OUTPUT_SVG.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    rows = load_rows(INPUT_CSV)
    summary = summarize(rows)
    write_summary_csv(summary)
    build_markdown(summary)
    write_svg(summary)
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_MD}")
    print(f"wrote {OUTPUT_SVG}")
    print(f"rows={len(rows)} groups={len(summary)}")


if __name__ == "__main__":
    main()
