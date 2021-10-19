import torch
from torch.nn import GRU, Dropout
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import BertModel, BertPreTrainedModel


class customModel(BertPreTrainedModel):
    def __init__(self, config, device):
        super(customModel, self).__init__(config)
        self.bert = BertModel(config)
        self.out_feature = self.bert.pooler.dense.out_features
        self.h0 = torch.randn(1, 1, self.in_feature, device=device)
        self.gru1 = GRU(1, 1, self.out_feature, bidirectional=True)
        self.gru2 = GRU(1, 1, self.out_feature, bidirectional=True)
        self.drop = Dropout(0.1)

    def forward(self, input):
        print(input)

        input_ids, attention_mask, return_type_ids = input
        output = self.bert(input_ids, attention_mask, return_type_ids)

        print(output.shape)

        pass


def init(model_args):
    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    return tokenizer, model
