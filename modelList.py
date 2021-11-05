from transformers import AutoConfig, AutoTokenizer
from model import RobertaQA, BertQA, ElectraQA, ConvModel


def init(model_args):
    # load config, tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )

    # insert speical tokens (unk, unused token)
    user_defined_symbols = []

    for i in range(1, 100):
        user_defined_symbols.append(f"[UNK{i}]")

    for i in range(500, 700):
        user_defined_symbols.append(f"[unused{i}]")

    special_tokens_dict = {"additional_special_tokens": user_defined_symbols}
    tokenizer.add_special_tokens(special_tokens_dict)

    # model_name_or_path마다 다른 모델을 불러오도록 작성되었습니다
    if model_args.model_name_or_path == "klue/bert-base":
        model = BertQA.from_pretrained(model_args.model_name_or_path, config=config)
    elif model_args.model_name_or_path == "klue/roberta-large":
        model = RobertaQA.from_pretrained(model_args.model_name_or_path, config=config)
    elif model_args.model_name_or_path == 'conv'
        model = ConvModel.from_pretrained('klue/roberta-large', config=config)
    elif model_args.model_name_or_path == "monologg/koelectra-base-v3-discriminator":
        model = ElectraQA.from_pretrained(model_args.model_name_or_path, config=config)

    # special token의 추가로 token embedding를 resize합니다
    model.resize_token_embeddings(tokenizer.vocab_size + len(user_defined_symbols))

    return tokenizer, model
