from transformers import AutoModelForSequenceClassification

def get_model(model_name, num_labels, len_tokenizer=None, trust_remote_code=False) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, trust_remote_code=trust_remote_code)
    if len_tokenizer:
        model.resize_token_embeddings(len_tokenizer)
        assert model.config.vocab_size == len_tokenizer,\
            f"Tokenizer size {len_tokenizer} does not match model size {model.config.vocab_size}"
    
    return model