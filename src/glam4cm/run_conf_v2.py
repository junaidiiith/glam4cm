import argparse
import os
import pandas as pd
import subprocess
from tqdm.auto import tqdm
from glam4cm.settings import (
    GRAPH_CLS_TASK,
    NODE_CLS_TASK,
    LINK_PRED_TASK,
    EDGE_CLS_TASK,
    results_dir
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str)
    args = parser.parse_args()
    return args


def get_embed_model_name(dataset_name, task_id, node_cls_label, edge_cls_label):
    if task_id == 6:
        label = f'LM_{GRAPH_CLS_TASK}/label'
    elif task_id == 7:
        label = f"LM_{NODE_CLS_TASK}/{node_cls_label}"
    elif task_id == 8:
        label = f"LM_{LINK_PRED_TASK}"
    elif task_id == 9:
        label = f"LM_{EDGE_CLS_TASK}/{edge_cls_label}"
        
    model_name = os.path.join(
        results_dir,
        dataset_name,
        label
    )
    
    return model_name

def execute_configs(run_configs, tasks_str: str):
    log_file = f"logs/run_configs_tasks_{tasks_str}.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=['Config', 'Status'])

    print(f"Total number of configurations in logs: {len(df)}")

    remaining_configs = [c for c in run_configs if c not in df['Config'].values.tolist()]

    for script_command in tqdm(remaining_configs, desc='Running tasks'):
        print(f'Running {script_command}')
        result = subprocess.run(f'python glam_test.py {script_command}', shell=True)

        status = 'success' if result.returncode == 0 else f'❌ {result.stderr}'
        print(f"✅ finished running command: {script_command}" if result.returncode == 0 else f"❌ failed with error:\n{result.stderr}")
        
        df.loc[len(df)] = [script_command, status]
        df.to_csv(log_file, index=False)


def get_run_configs(tasks):
    dataset_confs = {
        'ecore_555': {
            "node_cls_label": ["abstract"],
            "edge_cls_label": "type",
            "extra_params": {
                "num_epochs": 3,
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
            "bert_config": {
                "train_batch_size": 2,
            },
            "gnn_config": {
                "task_id": 6,
            },
        },
        3: {
            "bert_config": {
                "train_batch_size": 32,
            },
            "gnn_config": {
                "task_id": 7,
            },
        },
        4: {
            "bert_config": {
                "train_batch_size": 64,
            },
            "gnn_config": {
                "task_id": 8,
            },
        },
        5: {
            "bert_config": {
                "train_batch_size": 64,
            },
            "gnn_config": {
                "task_id": 9,
            },
        }
    }

    dataset_updates = [
        "",
        "use_attributes", 
        "use_node_types", 
        "use_edge_label", 
        "use_edge_types", 
        "use_special_tokens"
    ]

    gnn_conf = {
        "lr": 1e-3
    }

    gnn_updates = [
        "",
        "use_embeddings",
        "use_edge_attrs"   
    ]

    gnn_models = [
        {
            "name": "SAGEConv",
            "params": {}
        },
        {
            "name": "GATv2Conv",
            "params": {
                "num_heads": 4
            }
        }
    ]

    gnn_train = True

    run_configs = list()
    for task_id in tasks: 
        bert_task_config_str = f'--task_id={task_id} ' + ' '.join([f'--{k}={v}' for k, v in task_configs[task_id]['bert_config'].items()])
        
        for distance in range(4):
            distance_config_str = f' --distance={distance} '
            
            for i in range(len(dataset_updates)):
                config_task_str = ' '.join([f'--{u}' if u else '' for u in [x for x in dataset_updates[:i+1]]])
                    
                
                for dataset, dataset_conf in dataset_confs.items():
                    
                    if dataset == 'ontouml':
                        config_task_str = config_task_str.replace("--use_edge_label", "").replace("--use_edge_types", "")
                    
                    if (task_id == 2 and dataset not in ['ecore_555', 'modelset'])\
                        or (task_id in [4, 5] and dataset in ['ontouml']):
                        continue
                    
                    dataset_conf_str = f' --dataset={dataset} ' + ' '.join([f'--{k}={v}' for k, v in dataset_conf['extra_params'].items()]) + ' --min_edges=10 '
                    
                    node_cls_labels = dataset_conf['node_cls_label'] if isinstance(dataset_conf['node_cls_label'], list) else [dataset_conf['node_cls_label']]
                    edge_cls_labels = (dataset_conf['edge_cls_label'] if isinstance(dataset_conf['edge_cls_label'], list) else [dataset_conf['edge_cls_label']]) if 'edge_cls_label' in dataset_conf else []
                    for node_cls_label in node_cls_labels:
                        for edge_cls_label in edge_cls_labels:
                            labels_conf_str = f'--node_cls_label={node_cls_label} --edge_cls_label={edge_cls_label} '
                            
                            bert_config = f"{bert_task_config_str} {dataset_conf_str} {labels_conf_str} {config_task_str} {distance_config_str}"
                            
                            run_configs.append(bert_config)
                            
                            if gnn_train:
                                for gnn_model in gnn_models:
                                    for j in range(len((gnn_updates))):
                                        gnn_task_config_str = ' '.join([f'--{u}={v}' if u else '' for u, v in task_configs[task_id]['gnn_config'].items()])
                                        gnn_config_str = ' '.join([f'--{u}' if u else '' for u in [i for i in gnn_updates[:j+1]]])
                                        gnn_params_str = f' --gnn_conv_model={gnn_model["name"]} ' + ' '.join([f'--{k}={v}' for k, v in gnn_model['params'].items()]) + ' ' + ' '.join([f'--{k}={v}' for k, v in gnn_conf.items()]) + ' '
                                        
                                        if "use_embeddings" in gnn_updates[:j+1]:
                                            gnn_task_id = task_configs[task_id]['gnn_config']['task_id']
                                            gnn_params_str += f' --ckpt={get_embed_model_name(dataset, gnn_task_id, node_cls_label, edge_cls_label)} ' 
                                        
                                        run_configs.append(f"{gnn_task_config_str} {dataset_conf_str.replace(f"--num_epochs={dataset_conf['extra_params']['num_epochs']}", "--num_epochs=200")} {labels_conf_str} {distance_config_str} {config_task_str} {gnn_config_str} {gnn_params_str}")

    print(f"Total number of configurations: {len(run_configs)}")
    return run_configs


def main():
    tasks = [int(i) for i in get_args().tasks.split(',')]
    run_configs = get_run_configs(tasks)
    # Execute the configurations
    execute_configs(run_configs, tasks_str="_".join([str(i) for i in tasks]))
    # Save the configurations to a CSV file
        
if __name__ == '__main__':
    main()