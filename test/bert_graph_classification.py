from test.common_args import get_common_args_parser
from trainers.bert_classifier import train_hf
from data_loading.models_dataset import EcoreModelDataset 

def parse_args():
    parser = get_common_args_parser()
    return parser.parse_args()


def run(args):

    config_params = dict(
        timeout = args.timeout,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload=args.reload
    )
    dataset_name = args.dataset
    dataset = EcoreModelDataset(dataset_name, **config_params)

    model_name = args.model_name
    epochs = args.epochs
    train_hf(model_name, dataset, epochs=epochs)