from glam4cm.downstream_tasks.common_args import (
    get_bert_args_parser, 
    get_common_args_parser, 
)
import os
from glam4cm.data_loading.graph_dataset import GraphNodeDataset
from glam4cm.data_loading.models_dataset import get_models_dataset
from glam4cm.models.llm import LLMService
from glam4cm.settings import NODE_CLS_TASK, results_dir
from glam4cm.tokenization.special_tokens import *

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score,
    balanced_accuracy_score,
    precision_score,
)
import json

from glam4cm.tokenization.utils import get_tokenizer
from glam4cm.utils import merge_argument_parsers, set_encoded_labels, set_seed
from pydantic import BaseModel, Field


from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import os
from pydantic import BaseModel


from pydantic import BaseModel, Field

from glam4cm.data_loading.models_dataset import get_models_dataset
from glam4cm.data_loading.graph_dataset import GraphNodeDataset


PROMPTS_DATA_DIR = "prompts_data"


class ClassPrediction(BaseModel):
    node_id: str = Field(..., description="The ID of the class node being predicted")
    node_text: str = Field(..., description="The textual description of the class and its neighbours")
    prediction: str = Field(..., description="The predicted type of the class, which should be one of the possible class types")

class LLMNodeClassificationPrediction(BaseModel):
    predictions: list[ClassPrediction] = Field(..., description="A list of predictions for each class in the input")


system_prompt = \
"""
You are an expert in predicting the type semantics of a software model in Archimate and UML models. 
"""

user_prompt = \
"""
Below are the textual descriptions of {num_elements} model elements of a software model and its neighbouring classes.
You need to predict the type semantics of the each class based on the descriptions and the information of its neighbours.
The format of each class description is as follows:
node_id: <the ID of the class node being predicted>
Description:
<the textual description of the class and its neighbours spanning multiple lines>

Below is the list of all class descriptions for the classes being predicted:
{class_descriptions}

The possible class types are: 
{all_classes}

MAKE SURE THAT THE CLASS IS ONLY ONE OF THE ABOVE TYPES. DO NOT MAKE UP ANY OTHER CLASS TYPES.
"""

response_format = \
"""
The output should be in the following JSON format:
[
    {
        "node_data": "<the textual description of the class and its neighbours>",
        "predicted_type": "<the predicted type of the class, which should be one of the possible class types>"
    },
    ...
]
"""


def get_llm_prompts(
    graph_dataset: GraphNodeDataset, 
    distance: int = 1, 
    label: str = 'type', 
    batch_size: int = 20,
    classification_classes_str: str = None,
    user_prompt: str = user_prompt,
    system_prompt: str = system_prompt
):
    cls_data = graph_dataset.get_node_classification_texts(distance=distance, label=label)
    train_texts = cls_data['train_nodes']
    assert hasattr(graph_dataset, f'node_label_map_{label}'), f"Graph dataset does not have node_label_map for label {label}"
    
    node_label_map = getattr(graph_dataset, f'node_label_map_{label}')
    train_labels = node_label_map.inverse_transform(cls_data['train_node_classes'])
    test_texts = cls_data['test_nodes']
    test_labels = node_label_map.inverse_transform(cls_data['test_node_classes'])
    all_classes = list([c for c in node_label_map.classes_ if c])
    def construct_prompt(texts):
        class_descriptions = "\n\n".join([f"node_id: {i+1}\nDescription: \n{text}" for i, text in enumerate(texts)])
        all_classes_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(all_classes)]) if classification_classes_str is None else classification_classes_str
        return user_prompt.format(num_elements=len(texts), class_descriptions=class_descriptions, all_classes=all_classes_str)
    data = {
        "train_prompts": [
            [{"role": "system", "content": system_prompt},
            {"role": "user", "content": construct_prompt(train_texts[i:i+batch_size])}]
            for i in range(0, len(train_texts), batch_size)
        ],
        "test_prompts": [
            [{"role": "system", "content": system_prompt},
            {"role": "user", "content": construct_prompt(test_texts[i:i+batch_size])}]
            for i in range(0, len(test_texts), batch_size)
        ],
        "train_labels": [train_labels[i:i+batch_size] for i in range(0, len(train_labels), batch_size)],
        "test_labels": [test_labels[i:i+batch_size] for i in range(0, len(test_labels), batch_size)],
    }
    return data
    


def run_llm_node_classification(
    dataset_name: str, 
    graph_config: dict,
    llm_client: LLMService = LLMService(),
    max_workers: int = 5,
    classification_classes_str: str = None,
    user_prompt: str = user_prompt,
    system_prompt: str = system_prompt
):
    config_params = dict(
        min_edges = graph_config.get('min_edges', 10),
        min_enr = graph_config.get('min_enr', -1)
    )
    distance = graph_config.get('distance', 1)
    dataset = get_models_dataset(dataset_name, **config_params)
    graph_data_params = dict(
        task_type = 'node_cls',
        use_edge_label = graph_config.get('use_edge_label', True),
        use_node_types = graph_config.get('use_node_types', True),
        use_edge_types = graph_config.get('use_edge_types', True),
        node_topk = graph_config.get('node_topk', -1),
    )
    graph_dataset = GraphNodeDataset(
        dataset, 
        **graph_data_params
    )
    label = graph_config.get('node_cls_label', 'type')
    llm_prompts_dataset = get_llm_prompts(
        graph_dataset, 
        distance=distance, 
        label=label,
        classification_classes_str=classification_classes_str,
        user_prompt=user_prompt,
        system_prompt=system_prompt
    )
    responses = llm_client.get_llm_response_parallel(
        messages_list=llm_prompts_dataset['test_prompts'],
        response_format=LLMNodeClassificationPrediction,
        max_workers=max_workers,
        function_name=f"{dataset_name}_{label}_distance_{graph_config.get('distance', 1)}"
    )
    preds, acts = list(), list()
    for response, labels in zip(responses, llm_prompts_dataset['test_labels']):
        try:
            assert len(response['predictions']) == len(labels), \
            f"Number of predictions {len(response['predictions'])} does not match number of labels {len(labels)}"
            for pred, act in zip(response['predictions'], labels):
                preds.append(pred['prediction'])
                acts.append(act)
        except AssertionError as e:
            print(f"Error occurred: {e}")
    
    if classification_classes_str is not None:
        preds = [False if p == 'false' else True for p in preds]
    
    precision = precision_score(acts, preds, average='macro', zero_division=0)
    recall = recall_score(acts, preds, average='macro', zero_division=0)
    f1 = f1_score(acts, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(acts, preds)
    print(f"Results for {dataset_name} with distance {distance}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # "preds": preds,
        # "acts": acts
    }
    with open(f"{results_dir}/{dataset_name}_distance_{distance}_results.json", "w") as f:
        json.dump(result, f, indent=4)
    return result


def get_parser():
    common_parser = get_common_args_parser()
    bert_parser = get_bert_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)

    parser.add_argument('--oversampling_ratio', type=float, default=-1)

    return parser



def run(args):
    
    print("Training model")
    
    classes_str = "1. `true`: if Ecore class is an abstract class\n2. `false`: if Ecore class is a concrete class\n Make sure the predicted type is boolean either `true` or `false`"

    run_llm_node_classification(
        dataset_name=args.dataset,
        graph_config={
            "min_edges": args.min_edges,
            "use_node_types": args.use_node_types,
            "use_edge_types": args.use_edge_types,
            "use_edge_label": args.use_edge_label,
            "use_attributes": args.use_attributes,
            "distance": args.distance,
            "use_edge_label": args.use_edge_label,
            "node_cls_label": "abstract",
            "node_topk": args.node_topk
        },
        classification_classes_str=classes_str
    )
    

if __name__ == '__main__':
    args = get_parser()
    run(args)
