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
    parser.add_argument('--start', type=int, default=-1)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--run_lm', action='store_true')
    parser.add_argument('--run_gnn', action='store_true')
    parser.add_argument('--min_distance', type=int, default=0)
    parser.add_argument('--max_distance', type=int, default=3)
    parser.add_argument('--distances', type=str, default=None)
    
    args = parser.parse_args()
    return args

args = get_args()


dataset_confs = {
    'eamodelset': {
        "node_cls_label": ["type", "layer"],
        "edge_cls_label": "type",
        "extra_params": {
            "num_epochs": 50,
        }
    },
    'ecore_555': {
        "node_cls_label": ["abstract"],
        "edge_cls_label": "type",
        "extra_params": {
            "num_epochs": 50,
        }
    },
    'modelset': {
        "node_cls_label": ["abstract"],
        "edge_cls_label": "type",
        "extra_params": {
            "num_epochs": 50,
        }
    },
    'ontouml': {
        "node_cls_label": ["stereotype"],
        "edge_cls_label": "type",
        "extra_params": {
            "num_epochs": 50,
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
    },
    11: {
        "bert_config": {
            "train_batch_size": 1024,
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


def cmd_to_dict(command_line):
    return {
        i.split('=')[0].replace('--', ''): True if '=' not in i else i.split('=')[1] 
        for i in command_line.split()
    }


def get_config_str(command_line):
    args = cmd_to_dict(command_line)
    config_str = ""
    if 'use_attributes' in args:
        config_str += "_attrs"
    if 'use_edge_label' in args:
        config_str += "_el"
    if 'use_edge_types' in args:
        config_str += "_et"
    if 'use_node_types' in args:
        config_str += "_nt"
    if 'use_special_tokens' in args:
        config_str += "_st"
    if 'no_labels' in args:
        config_str += "_nolb"
    if "node_cls_label" in args:
        config_str += f"_{args['node_cls_label']}"
    if "edge_cls_label" in args:
        config_str += f"_{args['edge_cls_label']}"
    if "distance" in args:
        config_str += f"_{args['distance']}"

    return config_str



def get_embed_model_name(command_line):
    args = cmd_to_dict(command_line)
    task_id = int(args['task_id'])
    
    if task_id == 6:
        label = f'LM_{GRAPH_CLS_TASK}/label'
    elif task_id == 7:
        label = f"LM_{NODE_CLS_TASK}/{args['node_cls_label']}"
    elif task_id == 8:
        label = f"LM_{LINK_PRED_TASK}"
    elif task_id == 9:
        label = f"LM_{EDGE_CLS_TASK}/{args['edge_cls_label']}"
        
    model_name = os.path.join(
        results_dir,
        args['dataset'],
        label,
        get_config_str(command_line)
    )
    if not os.path.exists(model_name):
        print(model_name, os.path.exists(model_name), " does not exist")
    return model_name


def execute_configs(run_configs, tasks_str: str):
    
    log_file = f"logs/run_configs_tasks_{tasks_str}_cmgpt.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=['Config', 'Status'])
    remaining_configs = {c['lm']: c['gnn'] for c in run_configs if c['lm'] not in df['Config'].values}
    
    start = 0 if args.start == -1 else args.start
    end = len(remaining_configs) if args.end == -1 else args.end
    lm_script_commands = [lm_script_command for lm_script_command in remaining_configs.keys()][start:end]

    for lm_script_command in lm_script_commands:
        remaining_configs[lm_script_command] = [
            gnn_script_command + ' --ckpt=' + get_embed_model_name(gnn_script_command) 
            if 'use_embeddings' in gnn_script_command 
            else gnn_script_command
            for gnn_script_command in remaining_configs[lm_script_command]
        ]
    print("\n".join([r for r in remaining_configs]))
    print("Total number of configurations: ", len(run_configs))
    print(f"Total number of remaining configurations: {len(remaining_configs)}")
    print("Total number of configurations to run: ", len(remaining_configs) + sum([len(v) for v in remaining_configs.values()]))
    import json
    print(json.dumps(remaining_configs, indent=2), len(remaining_configs))
    
    for lm_script_command in tqdm(lm_script_commands, desc=f'Running tasks: {start}-{end-1}'):
        if args.run_lm:
            lm_script_command = lm_script_command.replace("train_batch_size", "batch_size")
            print(f'Running LM --> {lm_script_command}')
            result = subprocess.run(f'python glam_test.py {lm_script_command}', shell=True)

            status = 'success' if result.returncode == 0 else f'❌ {result.stderr}'
            print(f"✅ finished running command: {lm_script_command}" if result.returncode == 0 else f"❌ failed with error:\n{result.stderr}")
            
            df.loc[len(df)] = [lm_script_command, status]
            df.to_csv(log_file, index=False)
        
        if args.run_gnn:
            for gnn_script_command in tqdm(remaining_configs[lm_script_command], desc='Running GNN'):
                print(f'Running GNN --> {gnn_script_command}')
                
                result = subprocess.run(f'python glam_test.py {gnn_script_command}', shell=True)

                status = 'success' if result.returncode == 0 else f'❌ {result.stderr}'
                print(f"✅ finished running command: {gnn_script_command}" if result.returncode == 0 else f"❌ failed with error:\n{result.stderr}")

                df.loc[len(df)] = [gnn_script_command, status]
                df.to_csv(log_file, index=False)


def get_run_configs(tasks):

    run_configs = list()
    for task_id in tasks: 
        bert_task_config_str = [f'--task_id={task_id}'] + [f'--{k}={v}' for k, v in task_configs[task_id]['bert_config'].items()] + (['--reload'] if args.reload else [])
        
        if args.distances:
            distances = [int(i) for i in args.distances.split(',')]
        else:
            distances = [d for d in range(args.min_distance, args.max_distance + 1)]
            
        for distance in distances:
            distance_config_str = [f'--distance={distance}']
            
            for i in range(len(dataset_updates)):
                if i < len(dataset_updates) - 1:
                    continue
                
                for dataset, dataset_conf in dataset_confs.items():
                    if (task_id == 2 and dataset not in ['ecore_555', 'modelset'])\
                        or (task_id in [4, 5] and dataset in ['ontouml']):
                        continue
                    dataset_conf_str = [f'--dataset={dataset}'] + [f'--{k}={v}' for k, v in dataset_conf['extra_params'].items()] + ['--min_edges=10']
                    node_cls_labels = dataset_conf['node_cls_label'] if isinstance(dataset_conf['node_cls_label'], list) else [dataset_conf['node_cls_label']]
                    edge_cls_labels = (dataset_conf['edge_cls_label'] if isinstance(dataset_conf['edge_cls_label'], list) else [dataset_conf['edge_cls_label']]) if 'edge_cls_label' in dataset_conf else []
                    for node_cls_label in node_cls_labels:
                        for edge_cls_label in edge_cls_labels:
                            labels_conf_str = [f'--node_cls_label={node_cls_label}', f'--edge_cls_label={edge_cls_label}']
                            
                            config_task_str = [f'--{u}' if u else '' for u in [x for x in dataset_updates[:i+1]]]
                            # print(config_task_str)
                            # if dataset == 'eamodelset':
                            #     continue
                            if dataset == 'ontouml':
                                if "--use_edge_label" in config_task_str:
                                    config_task_str.remove("--use_edge_label")
                                    
                            if dataset == 'eamodelset':
                                if "--use_edge_label" in config_task_str:
                                    config_task_str.remove("--use_edge_label")
                                if "--use_attributes" in config_task_str:
                                    config_task_str.remove("--use_attributes")
                            
                            bert_config = " ".join(bert_task_config_str + \
                                dataset_conf_str + \
                                labels_conf_str + \
                                config_task_str + \
                                distance_config_str
                            )
                            
                            # if distance > 1:
                            #     bert_config = bert_config.replace(f"--train_batch_size={task_configs[task_id]['bert_config']['train_batch_size']}", "--train_batch_size=4")
                            # print(bert_config)
                            run_configs.append({'lm': bert_config})
                            
                            if gnn_train:
                                gnn_configs = list()
                                for gnn_model in gnn_models:
                                    for j in range(len((gnn_updates))):
                                        gnn_task_config_str = [f'--{u}={v}' if u else '' for u, v in task_configs[task_id]['gnn_config'].items()] + (['--reload'] if args.reload else [])
                                        gnn_config_str = [f'--{u}' if u else '' for u in [i for i in gnn_updates[:j+1]]]
                                        gnn_params_str = [f'--gnn_conv_model={gnn_model["name"]}'] + \
                                            [f'--{k}={v}' for k, v in gnn_model['params'].items()] + \
                                            [f'--{k}={v}' for k, v in gnn_conf.items()]
                                        
                                        gnn_config = " ".join(
                                            gnn_task_config_str + \
                                            gnn_config_str + \
                                            gnn_params_str + \
                                            dataset_conf_str + \
                                            labels_conf_str + \
                                            config_task_str + \
                                            distance_config_str
                                        )
                                        gnn_config = gnn_config.replace(f"--train_batch_size={task_configs[task_id]['bert_config']['train_batch_size']}", "--train_batch_size=8")
                                        gnn_config = gnn_config.replace(f"--num_epochs={dataset_conf['extra_params']['num_epochs']}", "--num_epochs=200")
                                        gnn_configs.append(gnn_config)
                                
                                run_configs[-1]['gnn'] = gnn_configs


    
    return run_configs


def get_remaining_configs(tasks_str, run_configs):

    def change_batch_size(conf: str):
        if "distance=2" in conf or "distance=3" in conf:
            conf.replace("--batch_size=64", "--batch_size=8")\
                .replace("--batch_size=32", "--batch_size=8")\
                .replace("--batch_size=16", "--batch_size=8")
        return conf

    
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/run_configs_tasks_{tasks_str}.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=['Config', 'Status'])
        return {
            'df': df,
            'configs': run_configs,
            'log_file': log_file
        }

    # your data
    v = df['Config'].apply(lambda x: int(x.split('--task_id=')[1].split()[0]))
    parent_child = {2: 6, 3: 7, 4: 8, 5: 9}
    parent_idxs = [i for i, val in enumerate(v) if val in parent_child]

    mapping = []
    
    for idx, pidx in enumerate(parent_idxs):
        start = pidx + 1
        end   = parent_idxs[idx+1] if idx+1 < len(parent_idxs) else len(v)

        child_val = parent_child[v[pidx]]
        children = [j for j in range(start, end) if v[j] == child_val]

        mapping.append({pidx: children})

    rem_configs = dict()
    to_delete = []
    num_gnn = 0
    for mapping in mapping:
        lm_idx = list(mapping.keys())[0]
        gnn_indices = mapping[lm_idx]
        lm_config = change_batch_size(df.iloc[lm_idx]['Config'])
        if any(df.iloc[i]['Status'] != "success" for i in gnn_indices):
            rem_configs[lm_config] = [df.iloc[i]['Config'] for i in gnn_indices]
            num_gnn = len(rem_configs[lm_config])
            to_delete.append(lm_idx)
            to_delete.extend(gnn_indices)
    print(f"Total number of configurations to run: {len(rem_configs)*num_gnn}")
    df = df.drop(to_delete)
    df = df.reset_index(drop=True)
    
    return {
        'df': df,
        'configs': rem_configs,
        'log_file': log_file
    }


def main():
    tasks = [int(i) for i in args.tasks.split(',')]
    
    run_configs = get_run_configs(tasks)
    # Execute the configurations
    execute_configs(run_configs, tasks_str="_".join([str(i) for i in tasks]))
    # Save the configurations to a CSV file
        
if __name__ == '__main__':
    main()