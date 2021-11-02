from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from model import lstmOnRoberta, lstmOnRoberta2


def init(ver, model_args, training_args, dropout):
    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )
    config.num_labels = 2
    config.dropout = dropout

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )

    # insert speical tokens (unk, unused token)
    user_defined_symbols = []
    
    for i in range(1,10):
        user_defined_symbols.append(f'[UNK{i}]')

    for i in range(500,700):
        user_defined_symbols.append(f'[unused{i}]')
    
    special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
    tokenizer.add_special_tokens(special_tokens_dict)

    if ver=="original": model = AutoModelForQuestionAnswering.from_pretrained('klue/roberta-large', config=config)
    elif ver=="ver_1": model = lstmOnRoberta(dropout=dropout)
    else: model = lstmOnRoberta2(training_args, dropout=dropout)

    model.roberta.resize_token_embeddings(tokenizer.vocab_size + len(user_defined_symbols))

    return tokenizer, model
