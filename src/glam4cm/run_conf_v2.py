import argparse
import subprocess
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    tasks = [int(i) for i in get_args().tasks.split(',')]

    dataset_confs = {
        'ecore_555': {
            "node_cls_label": ["abstract"],
            "edge_cls_label": "type",
            "extra_params": {
                "num_epochs": 10,
            }
        },
        'modelset': {
            "node_cls_label": ["abstract"],
            "edge_cls_label": "type",
            "extra_params": {
                "num_epochs": 10,
            }
        },
        'eamodelset': {
            "node_cls_label": ["type", "layer"],
            "edge_cls_label": "type",
            "extra_params": {
                "num_epochs": 15,
            }
        },
        'ontouml': {
            "node_cls_label": ["stereotype"],
            "edge_cls_label": "type",
            "extra_params": {
                "num_epochs": 15,
                'node_topk': 20
            }
        },
    }

    task_configs = {
        2: {
            "train_batch_size": 2,
        },
        3: {
            "train_batch_size": 32,
        },
        4: {
            "train_batch_size": 64,
        },
        5: {
            "train_batch_size": 64,
        }
    }

    updates = [
        "",
        "use_attributes", 
        "use_node_types", 
        "use_edge_label", 
        "use_edge_types", 
        "use_special_tokens"
    ]

    run_configs = list()
    for task_id in tasks: 
        for dataset, dataset_conf in dataset_confs.items():
            if task_id == 2 and dataset not in ['ecore_555', 'modelset']\
                or task_id in [4, 5] and dataset in ['ontouml']:
                continue
            
            current_update_idx = 0
            task_str = f'--task_id={task_id} --dataset={dataset} ' + ' '.join([f'--{k}={v}' for k, v in dataset_conf['extra_params'].items()]) + ' '
            
            node_cls_labels = dataset_conf['node_cls_label'] if isinstance(dataset_conf['node_cls_label'], list) else [dataset_conf['node_cls_label']]
            edge_cls_labels = (dataset_conf['edge_cls_label'] if isinstance(dataset_conf['edge_cls_label'], list) else [dataset_conf['edge_cls_label']]) if 'edge_cls_label' in dataset_conf else []
            
            for distance in range(4):
                # if distance in [1, 2, 3] or dataset == 'ontouml':
                # if dataset != 'ontouml':
                #     continue
                for node_cls_label in node_cls_labels:
                    for edge_cls_label in edge_cls_labels:
                        for i in range(len(updates)):
                            current_update_idx = i
                            config_task_str = task_str + f'--node_cls_label={node_cls_label} --edge_cls_label={edge_cls_label} --min_edges=10 '
                            config_task_str += ' '.join([f'--{u}' if u else '' for u in [i for i in updates[:current_update_idx+1]]])
                            config_task_str += f' --distance={distance} '
                            config_task_str += ' '.join([f'--{k}={v}' for k, v in task_configs[task_id].items()])
                            run_configs.append(config_task_str)

    # run_configs = [c for c in run_configs if 'use_special_tokens' in c]

    for script_command in tqdm(run_configs, desc='Running tasks'):
        print(f'Running {script_command}')
        subprocess.run(f'python src/glam4cm/run.py {script_command}', shell=True)
