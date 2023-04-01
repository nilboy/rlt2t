import torch
import os
import tempfile
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSeq2SeqLM, \
                         AutoModelForCausalLM
from transformers import BertTokenizer

base_dir = os.path.dirname(os.path.dirname(__file__))

vocab_file = os.path.join(base_dir, 'vocab/vocab.txt')


class ModelConverter(object):
    def __init__(self, vocab_size=7200):
        self.vocab_size = vocab_size
        all_words = []
        with open(vocab_file) as fin:
            for line in fin:
                all_words.append(line.strip())
        # 创建不自动删除的临时文本文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            for word in all_words[0:vocab_size]:
                tmp_file.write(word + '\n')
        self.tokenizer = BertTokenizer(tmp_file.name)
        os.remove(tmp_file.name)

    def convert_bert_model(self,
                           input_model_name,
                           output_model_name, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(input_model_name)
        model = AutoModelForMaskedLM.from_pretrained(input_model_name)

        new_embedding_weight = model.bert.get_input_embeddings().weight[0:vocab_size].detach()
        new_cls_weight = model.cls.predictions.decoder.weight[0:vocab_size].detach()
        new_cls_bias = model.cls.predictions.bias.data[0:vocab_size].detach()

        model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
        model.cls.predictions.decoder = nn.Linear(in_features=model.config.hidden_size,
                                                  out_features=vocab_size, bias=True)
        model.cls.predictions.decoder.weight.data = new_cls_weight
        model.cls.predictions.bias = nn.Parameter(new_cls_bias)
        model.config.vocab_size = vocab_size
        model.half()
        model.save_pretrained(output_model_name)
        self.tokenizer.save_pretrained(output_model_name)

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
        self.tokenizer.save_pretrained(output_model_name)
        generate_config_file = os.path.join(output_model_name, 'generation_config.json')
        os.system(fr"rm {generate_config_file}")

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

        self.tokenizer.save_pretrained(output_model_name)
        generate_config_file = os.path.join(output_model_name, 'generation_config.json')
        os.system(fr"rm {generate_config_file}")
