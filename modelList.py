from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from model import RobertaQA, BertQA


def init(model_args):
    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    # insert speical tokens (unk, unused token)
    user_defined_symbols = []
    
    for i in range(1,100): user_defined_symbols.append(f'[UNK{i}]')

    for i in range(500,700): user_defined_symbols.append(f'[unused{i}]')
    
    special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
    tokenizer.add_special_tokens(special_tokens_dict)

    if model_args.model_name_or_path == "klue/bert-base":
        model = BertQA.from_pretrained(model_args.model_name_or_path, config=config)
    elif model_args.model_name_or_path == "klue/roberta-large":
        model = RobertaQA.from_pretrained(model_args.model_name_or_path, config=config)

    model.resize_token_embeddings(tokenizer.vocab_size + len(user_defined_symbols))

    return tokenizer, model
