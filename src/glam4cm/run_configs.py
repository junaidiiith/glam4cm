import argparse
import itertools
import subprocess
from tqdm.auto import tqdm



all_tasks = {
    1: [
        '--dataset=ecore_555 --num_epochs=5 --train_batch_size=2',
        '--dataset=modelset --num_epochs=10 --train_batch_size=2',
    ],
    
    2: [
        '--min_edges=10 --train_batch_size=2 --num_epochs=5',
    ],
    
    3: [
        '--min_edges=10 --train_batch_size=64 --distance=1 --num_epochs=10',
    ],
    
    4: [
        "--min_edges=10 --distance=1 --train_batch_size=64  --num_epochs=5",
    ],
    
    5: [
        "--min_edges=10 --distance=1 --train_batch_size=64  --num_epochs=5",
    ],
}

dataset_confs = {
    'ecore_555': {
        "node_cls_label": ["abstract"],
        "edge_cls_label": "type",
    },
    'modelset': {
        "node_cls_label": ["abstract"],
        "edge_cls_label": "type",
    },
    # 'mar-ecore-github': {
    #     "node_cls_label": ["abstract"],
    #     "edge_cls_label": "type",
    # },
    'eamodelset': {
        "node_cls_label": ["type", "layer"],
        "edge_cls_label": "type",
    },
}

param_configs = {
    'use_attributes': [0, 1],
    'use_node_types': [0, 1],
    'use_edge_types': [0, 1],
    'use_edge_label': [0, 1],
}

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--tasks', type=str)
	args = parser.parse_args()
	return args


tasks = [int(i) for i in get_args().tasks.split(',')]
run_configs = list()

for task_id in all_tasks:
    if task_id not in tasks:
        continue
    
    for task_str in all_tasks[task_id]:    
        for dataset, dataset_conf in dataset_confs.items():
            if task_id == 2 and dataset not in ['ecore_555', 'modelset']:
                continue
            task_str = f'--task_id={task_id} ' + task_str
                
            node_cls_label = dataset_conf['node_cls_label'] if isinstance(dataset_conf['node_cls_label'], list) else [dataset_conf['node_cls_label']]
            edge_cls_label = dataset_conf['edge_cls_label'] if isinstance(dataset_conf['edge_cls_label'], list) else [dataset_conf['edge_cls_label']]
            
            for node_cls_label, edge_cls_label in itertools.product(node_cls_label, edge_cls_label):
                for params in itertools.product(*param_configs.values()):
                    config = {k: v for k, v in zip(param_configs.keys(), params)}
                    config_task_str = task_str + f' --dataset={dataset} --node_cls_label={node_cls_label} --edge_cls_label={edge_cls_label} ' + ' '.join([f'--{k}' if v else '' for k, v in config.items()])
                    # print(config_task_str)
                    run_configs.append(config_task_str)
                                        

for script_command in tqdm(run_configs, desc='Running tasks'):
    print(f'Running {script_command}')
    subprocess.run(f'python src/glam4cm/run.py {script_command}', shell=True)
