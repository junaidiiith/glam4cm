from tokenization.special_tokens import (
    EDGE_START, EDGE_END, NODE_SEP, NODE_PATH_SEP
)
from transformers import AutoTokenizer


def get_special_tokens():
    return {
        'additional_special_tokens': [EDGE_START, EDGE_END, NODE_SEP, NODE_PATH_SEP]
    }


def get_tokenizer(model_name, special_tokens, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.model_max_length = max_length
    return tokenizer
