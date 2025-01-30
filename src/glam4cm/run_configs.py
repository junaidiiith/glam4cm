import subprocess

from tqdm.auto import tqdm


tasks = {
    0: 'Create Dataset',

    1: 'BERT Graph Classification Comparison',
    2: 'BERT Graph Classification',
	3: 'BERT Node Classification',
    4: 'BERT Link Prediction',
    5: 'BERT Edge Classification',
    
    
    6: 'GNN Graph Classification',
    7: 'GNN Node Classification',
    8: 'GNN Edge Classification',
    9: 'GNN Link Prediction',
}

all_tasks = {
    1: [
        '--dataset=ecore_555 --num_epochs=5 --train_batch_size=2',
        '--dataset=modelset --num_epochs=10 --train_batch_size=2',
	],
    
	2: [
		'--dataset=ecore_555 --num_epochs=5 --min_edges=10 --train_batch_size=2',
        '--dataset=ecore_555 --num_epochs=5 --use_attributes --min_edges=10 --train_batch_size=2',
        '--dataset=ecore_555 --num_epochs=5 --use_edge_types --min_edges=10 --train_batch_size=2',
        '--dataset=ecore_555 --num_epochs=5 --use_attributes --use_edge_types --min_edges=10 --train_batch_size=2',
		'--dataset=modelset --num_epochs=10 --min_edges=10 --train_batch_size=2',
        '--dataset=modelset --num_epochs=10 --use_attributes --min_edges=10 --train_batch_size=2',
        '--dataset=modelset --num_epochs=10 --use_edge_types --min_edges=10 --train_batch_size=2',
        '--dataset=modelset --num_epochs=10 --use_attributes --use_edge_types --min_edges=10 --train_batch_size=2',
	],
    
	3: [
		'--dataset=ecore_555 --num_epochs=5 --cls_label=abstract --min_edges=10 --train_batch_size=32',
        '--dataset=ecore_555 --num_epochs=5 --use_attributes --cls_label=abstract --train_batch_size=32 --min_edges=10',
        '--dataset=ecore_555 --num_epochs=5 --use_edge_types --cls_label=abstract --train_batch_size=32 --min_edges=10',
        '--dataset=ecore_555 --num_epochs=5 --use_attributes --use_edge_types --cls_label=abstract --train_batch_size=32 --min_edges=10',
		'--dataset=modelset --num_epochs=10 --cls_label=abstract --train_batch_size=32 --min_edges=10',
        '--dataset=modelset --num_epochs=10 --use_attributes --cls_label=abstract --train_batch_size=32 --min_edges=10',
        '--dataset=modelset --num_epochs=10 --use_edge_types --cls_label=abstract --train_batch_size=32 --min_edges=10',
        '--dataset=modelset --num_epochs=10 --use_attributes --use_edge_types --cls_label=abstract --train_batch_size=32 --min_edges=10',
		
        '--dataset=mar-ecore-github --num_epochs=10 --use_attributes --use_edge_types --cls_label=abstract --train_batch_size=32 --min_edges=10',
        
		'--dataset=eamodelset --num_epochs=15 --cls_label=type --train_batch_size=32 --min_edges=10',
        '--dataset=eamodelset --num_epochs=15 --use_edge_types --cls_label=type --train_batch_size=32 --min_edges=10',
        '--dataset=eamodelset --num_epochs=15 --cls_label=layer --train_batch_size=32 --min_edges=10',
        '--dataset=eamodelset --num_epochs=15 --use_edge_types --cls_label=layer --train_batch_size=32 --min_edges=10',
	],
    
	4: [
		'--dataset=ecore_555 --num_epochs=3 --train_batch_size=32 --min_edges=10',
		'--dataset=ecore_555 --num_epochs=3 --use_attributes --train_batch_size=32 --min_edges=10',
		'--dataset=modelset --num_epochs=5 --train_batch_size=32 --min_edges=10 --reload',
		'--dataset=modelset --num_epochs=5 --use_attributes --train_batch_size=32 --min_edges=10 --reload',
		
        '--dataset=mar-ecore-github --num_epochs=5 --use_attributes --train_batch_size=32 --min_edges=10 --reload',
		'--dataset=eamodelset --num_epochs=5 --train_batch_size=32 --min_edges=10 --reload',
	],
	
    5: [
        '--dataset=ecore_555 --num_epochs=5 --train_batch_size=32 --min_edges=10 --reload',
        '--dataset=ecore_555 --num_epochs=5 --use_attributes --train_batch_size=32 --min_edges=10 --reload',
        '--dataset=modelset --num_epochs=10 --train_batch_size=32 --min_edges=10 --reload',
        '--dataset=modelset --num_epochs=10 --use_attributes --train_batch_size=32 --min_edges=10 --reload',
        '--dataset=mar-ecore-github --num_epochs=10 --use_attributes --train_batch_size=32 --min_edges=10 --reload',
        '--dataset=eamodelset --num_epochs=15 --train_batch_size=32 --min_edges=10 --reload',
    ],
	6: [
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --ckpt=results/ecore_555/graph_cls_/10_att_0_nt_0/checkpoint-225 --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --use_attributes --ckpt=results/ecore_555/graph_cls_/10_att_1_nt_0/checkpoint-225 --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --use_edge_types --ckpt=results/ecore_555/graph_cls_/10_att_0_nt_1/checkpoint-225 --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --use_attributes --use_edge_types --ckpt=results/ecore_555/graph_cls_/10_att_1_nt_1/checkpoint-225 --reload',
		
        '--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --ckpt=results/modelset/graph_cls_/10_att_0_nt_0/checkpoint-2540 --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --use_attributes --ckpt=results/modelset/graph_cls_/10_att_1_nt_0/checkpoint-2540 --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --use_edge_types --ckpt=results/modelset/graph_cls_/10_att_0_nt_1/checkpoint-2540 --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --use_embeddings --use_attributes --use_edge_types --ckpt=results/modelset/graph_cls_/10_att_1_nt_1/checkpoint-2540 --reload',
    ],
	7: [
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --ckpt=results/ecore_555/node_cls/abstract/abstract_10_att_0_nt_0/checkpoint-540 --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_attributes --ckpt=results/ecore_555/node_cls/abstract/abstract_10_att_1_nt_0/checkpoint-540 --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_edge_types --ckpt=results/ecore_555/node_cls/abstract/abstract_10_att_0_nt_1/checkpoint-540 --reload',
		'--dataset=ecore_555 --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_attributes --use_edge_types --ckpt=results/ecore_555/node_cls/abstract/abstract_10_att_1_nt_1/checkpoint-540 --reload',
		
		
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --ckpt=results/modelset/node_cls/abstract/abstract_10_att_0_nt_0/checkpoint-6870 --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_attributes --ckpt=results/modelset/node_cls/abstract/abstract_10_att_1_nt_0/checkpoint-6870 --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_edge_types --ckpt=results/modelset/node_cls/abstract/abstract_10_att_0_nt_1/checkpoint-6870 --reload',
		'--dataset=modelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_attributes --use_edge_types --ckpt=results/modelset/node_cls/abstract/abstract_10_att_1_nt_1/checkpoint-6870 --reload',
		
        '--dataset=mar-ecore-github --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --reload',
		'--dataset=mar-ecore-github --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --ckpt=results/mar-ecore-github/node_cls/abstract/abstract_10_att_0_nt_0/checkpoint-19400 --reload',
		'--dataset=mar-ecore-github --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_attributes --ckpt=results/mar-ecore-github/node_cls/abstract/abstract_10_att_1_nt_0/checkpoint-19400 --reload',
		'--dataset=mar-ecore-github --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_edge_types --ckpt=results/mar-ecore-github/node_cls/abstract/abstract_10_att_0_nt_1/checkpoint-19400 --reload',
		'--dataset=mar-ecore-github --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=abstract --use_embeddings --use_attributes --use_edge_types --ckpt=results/mar-ecore-github/node_cls/abstract/abstract_10_att_1_nt_1/checkpoint-19400 --reload',

        '--dataset=eamodelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=type --reload',
		'--dataset=eamodelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=type --use_embeddings --ckpt=results/eamodelset/node_cls/layer/layer_10_att_0_nt_0/checkpoint-9570 --reload',
		'--dataset=eamodelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=type --use_embeddings --use_edge_types --ckpt=results/eamodelset/node_cls/layer/layer_10_att_0_nt_1/checkpoint-9570 --reload',

        '--dataset=eamodelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=type --reload',
		'--dataset=eamodelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=type --use_embeddings --ckpt=results/eamodelset/node_cls/type/type_10_att_0_nt_0/checkpoint-9570 --reload',
		'--dataset=eamodelset --num_epochs=200 --batch_size=32 --min_edges=10 --cls_label=type --use_embeddings --use_edge_types --ckpt=results/eamodelset/node_cls/type/type_10_att_0_nt_1/checkpoint-9570 --reload',    
	]
}

allowed_tasks = [7]

for script_id in tqdm(allowed_tasks, desc='Running tasks'):
	task = tasks[script_id]
	for script in tqdm(all_tasks[script_id], desc=f'Running scripts for {task}'):
		script += f' --task={script_id} '
		print(f'Running {script}')

		subprocess.run(f'python run.py {script}', shell=True)
