from transformers import AutoModelForSequenceClassification

def get_model(model_name, num_labels, len_tokenizer=None) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if len_tokenizer:
        model.resize_token_embeddings(len_tokenizer)
        assert model.config.vocab_size == len_tokenizer,\
            f"Tokenizer size {len_tokenizer} does not match model size {model.config.vocab_size}"
    
    return model