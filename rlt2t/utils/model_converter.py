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

        new_embedding_weight = model.bert.get_input_embeddings().weight[0:vocab_size].detach()
        new_cls_weight = model.cls.predictions.decoder.weight[0:vocab_size].detach()
        new_cls_bias = model.cls.predictions.bias.data.detach()

        model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
        model.cls.predictions.decoder = nn.Linear(in_features=model.config.hidden_size,
                                                  out_features=vocab_size, bias=True)
        model.cls.predictions.decoder.weight.data = new_cls_weight
        model.cls.predictions.bias = nn.Parameter(new_cls_bias)
        model.config.vocab_size = vocab_size
        model.half()
        model.save_pretrained(output_model_name)
        tokenizer.save_pretrained(output_model_name)

    def convert_t5_model(self,
                         input_model_name,
                         output_model_name, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(input_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(input_model_name)

        new_embedding_weight = model.shared.weight[0:vocab_size].detach()
        new_lm_weight = model.lm_head.weight.data[0:vocab_size].detach()

        model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
        model.lm_head = nn.Linear(in_features=model.config.hidden_size,
                                  out_features=vocab_size, bias=False)
        model.lm_head.weight.data = new_lm_weight
        model.config.vocab_size = vocab_size
        model.half()
        model.save_pretrained(output_model_name)
        tokenizer.save_pretrained(output_model_name)

    def convert_gpt2_model(self,
                           input_model_name,
                           output_model_name, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(input_model_name)
        model = AutoModelForCausalLM.from_pretrained(input_model_name)

        new_embedding_weight = model.transformer.wte.weight[0:vocab_size].detach()
        new_lm_weight = model.lm_head.weight.data[0:vocab_size].detach()

        model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
        model.lm_head = nn.Linear(in_features=model.config.hidden_size,
                                  out_features=vocab_size, bias=False)
        model.lm_head.weight.data = new_lm_weight
        model.config.vocab_size = vocab_size
        model.half()
        model.save_pretrained(output_model_name)
        tokenizer.save_pretrained(output_model_name)
