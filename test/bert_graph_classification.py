from test.common_args import get_common_args_parser
from test.utils import get_models_dataset
from trainers.bert_classifier import train_hf


def get_parser():
    parser = get_common_args_parser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
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
    dataset = get_models_dataset(dataset_name, **config_params)

    model_name = args.model_name
    epochs = args.num_epochs
    train_hf(model_name, dataset, epochs=epochs)