from glam4cm.run import main

main()


# import glam4cm.utils
# from glam4cm.data_loading.graph_dataset import GraphNodeDataset, GraphEdgeDataset
# from glam4cm.settings import NODE_CLS_TASK, EDGE_CLS_TASK, LINK_PRED_TASK, GRAPH_CLS_TASK
# from transformers import AutoTokenizer
# from glam4cm.data_loading.models_dataset import EcoreDataset, OntoUMLDataset

# config_params = dict(
#     reload=True,
    
#     min_enr = -1,
#     min_edges = 10,
#     # language = 'en',
# )

# # dataset = ArchiMateDataset('eamodelset', **config_params)
# dataset = EcoreDataset('modelset', **config_params)
# # dataset = OntoUMLDataset('ontouml', **config_params)

# glam4cm.utils.set_seed(42)

# updates = ["", "use_attributes", "use_node_types", "use_edge_label", "use_edge_types", "use_special_tokens"]
# task_types = [LINK_PRED_TASK, GRAPH_CLS_TASK]
# for task_type in task_types:
#     graph_data_params = dict(
#         task_type=task_type,
#         # reload=True,
#         test_ratio=0.2,
#         # add_negative_train_samples=True,
#         # neg_sampling_ratio=1,
#         distance=1,
#         random_embed_dim=128,
#         use_attributes=True,
#         use_edge_label=True,
#         use_edge_types=True,
#         use_node_types=True,

#         use_special_tokens=True,
#         # task_type='graph_cls',
#         # use_embeddings=True,
#         # embed_model_name='bert-base-cased',
#         # ckpt='results/eamodelset/lp/10_att_0_nt_0/checkpoint-177600',
#     )
#     # for i, update in enumerate(updates):
#     #     if i != 0:
#     #         graph_data_params.update({update: True})
#     print(graph_data_params)
#     print("Loading graph dataset")
#     if task_type in [LINK_PRED_TASK, EDGE_CLS_TASK]:
#         graph_data_params = {**graph_data_params, 'add_negative_train_samples': True, 'neg_sampling_ratio': 1, 'edge_cls_label': 'type'}
#         graph_dataset = GraphEdgeDataset(dataset, **graph_data_params)
        
#     else:
#         graph_data_params = {**graph_data_params, 'node_cls_label': 'abstract'}
#         graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
#     print("Loaded graph dataset")
    
#     if task_type in [NODE_CLS_TASK]:
#         graph_dataset.get_node_classification_texts(distance=1, label='abstract')
#     elif task_type in [EDGE_CLS_TASK, LINK_PRED_TASK]:
#         graph_dataset.get_link_prediction_texts()
    
#     elif task_type in [GRAPH_CLS_TASK]:
#         graph_dataset.get_lm_graph_classification_data(AutoTokenizer.from_pretrained('bert-base-cased'))