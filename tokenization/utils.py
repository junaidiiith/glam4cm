from re import finditer
from tokenization.special_tokens import (
    EDGE_START, EDGE_END, NODE_SEP, NODE_PATH_SEP, NODE_LABEL
)
from transformers import AutoTokenizer

def get_special_tokens():
    return {
        'additional_special_tokens': [EDGE_START, EDGE_END, NODE_SEP, NODE_PATH_SEP, NODE_LABEL]
    }


def get_tokenizer(model_name, special_tokens=False, max_length=512) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if special_tokens:
        tokenizer.add_special_tokens(get_special_tokens())

    tokenizer.model_max_length = max_length
    return tokenizer


def camel_case_split(identifier) -> list:
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def doc_tokenizer(doc, lower=False) -> str:
    words = doc.split()
    # split _
    words = [w2 for w1 in words for w2 in w1.split('_') if w2 != '']
    # camelcase
    words = [w2.lower() if lower else w2 for w1 in words for w2 in camel_case_split(w1) if w2 != '']
    return " ".join(words)
