import json
import os


CONFIG_FILE_NAME = "config.json"
RESULTS_FILE_NAME = "results.json"
TENSORBOARD_LOGS_FILE_NAME = "tensorboard_logs.json"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=4, default=_json_default)


def get_experiment_dir(args, task_type, cls_label, config_str):
    """
    Create the canonical downstream result directory for one run config.

    Each task supplies its model-specific task type, for example
    ``LM_node_cls`` or ``GNN_node_cls``. This keeps BERT and GNN artifacts
    separate while keeping their result layout identical.
    """
    dataset = getattr(args, "dataset", getattr(args, "dataset_name", "default"))
    experiment_dir = os.path.join(
        getattr(args, "results_dir", "results"),
        task_type,
        dataset,
        str(cls_label or "default"),
        config_str or "default",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    _write_json(os.path.join(experiment_dir, CONFIG_FILE_NAME), vars(args))
    return experiment_dir


def get_finetuned_model_dir(args, task_type, cls_label, config_id):
    dataset = getattr(args, "dataset", getattr(args, "dataset_name", "default"))
    model_dir = os.path.join(
        getattr(args, "models_dir", "ftlm"),
        task_type,
        dataset,
        str(cls_label or "default"),
        config_id or "default",
    )
    os.makedirs(model_dir, exist_ok=True)
    _write_json(os.path.join(model_dir, CONFIG_FILE_NAME), vars(args))
    return model_dir


def export_tensorboard_logs(log_dir, output_path=None):
    """
    Export all TensorBoard scalar summaries below ``log_dir`` to JSON.

    TensorBoard stores summaries inside event files. Downstream trainers in
    this package log scalars, so the exported JSON groups every scalar point
    by tag and preserves the source event file, step, wall time, and value.
    """
    logs = {}
    for root, _, files in os.walk(log_dir):
        for file_name in sorted(files):
            if not file_name.startswith("events.out.tfevents"):
                continue
            from glam4cm.utils import parse_event_file

            event_path = os.path.join(root, file_name)
            event_source = os.path.relpath(event_path, log_dir)
            for event in parse_event_file(event_path):
                if not event.summary:
                    continue
                for summary in event.summary.value:
                    if not summary.HasField("simple_value"):
                        continue
                    logs.setdefault(summary.tag, []).append({
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": summary.simple_value,
                        "event_file": event_source,
                    })

    if output_path is not None:
        _write_json(output_path, logs)
    return logs


def save_experiment_results(experiment_dir, results=None):
    if results is not None:
        _write_json(os.path.join(experiment_dir, RESULTS_FILE_NAME), results)
    return export_tensorboard_logs(
        experiment_dir,
        os.path.join(experiment_dir, TENSORBOARD_LOGS_FILE_NAME),
    )


def get_logging_steps(dataset_size, num_epochs, batch_size):
    """
    Calculate the logging steps based on the dataset size, number of epochs, and batch size.
    """
    num_steps = dataset_size // batch_size
    logging_steps = num_steps * num_epochs // 20
    print(f"Logging steps: {logging_steps}")
    return logging_steps
