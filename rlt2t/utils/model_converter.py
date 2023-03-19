import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSeq2SeqLM, \
                         AutoModelForCausalLM

class ModelConverter(object):
    def __init__(self, vocab_size=7200):
        self.vocab_size = vocab_size

    def convert_bert_model(self,
                           input_model_name,
                           output_model_name, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(input_model_name)
        model = AutoModelForMaskedLM.from_pretrained(input_model_name)
        new_embedding_weight = torch.normal(0.0, 1.0, size=(vocab_size, model.config.hidden_size))
        model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
        model.cls.predictions.decoder = nn.Linear(in_features=model.config.hidden_size,
                                                  out_features=vocab_size, bias=True)
        model.cls.predictions.bias = nn.Parameter(torch.normal(0, 1.0, size=(vocab_size,)))
        model.config.vocab_size = vocab_size
        model.save_pretrained(output_model_name)
        tokenizer.save_pretrained(output_model_name)

    def convert_t5_model(self,
                         input_model_name,
                         output_model_name, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(input_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(input_model_name)
        new_embedding_weight = torch.normal(0.0, 1.0, size=(vocab_size, model.config.hidden_size))
        model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
        model.lm_head = nn.Linear(in_features=model.config.hidden_size,
                                  out_features=vocab_size, bias=False)
        model.config.vocab_size = vocab_size
        model.save_pretrained(output_model_name)
        tokenizer.save_pretrained(output_model_name)

    def convert_gpt2_model(self,
                           input_model_name,
                           output_model_name, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(input_model_name)
        model = AutoModelForCausalLM.from_pretrained(input_model_name)
        new_embedding_weight = torch.normal(0.0, 1.0, size=(vocab_size, model.config.hidden_size))
        model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
        model.lm_head = nn.Linear(in_features=model.config.hidden_size,
                                  out_features=vocab_size, bias=False)
        model.config.vocab_size = vocab_size
        model.save_pretrained(output_model_name)
        tokenizer.save_pretrained(output_model_name)
