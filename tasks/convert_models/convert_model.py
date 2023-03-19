import fire
from rlt2t.utils.model_converter import ModelConverter


def convert_model(input_model_name,
                  output_model_name,
                  model_type, vocab_size):
    model_convert = ModelConverter(vocab_size)
    if model_type == 'bert':
        model_convert.convert_bert_model(input_model_name, output_model_name)
    elif model_type == 't5':
        model_convert.convert_t5_model(input_model_name, output_model_name)
    elif model_type == 'gpt2':
        model_convert.convert_gpt2_model(input_model_name, output_model_name)
    else:
        raise ValueError(f'not support model_type: {model_type}')


if __name__ == '__main__':
    fire.Fire(convert_model)