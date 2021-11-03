import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, CrossEntropyLoss, Dropout

from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class lstmOnRoberta(RobertaPreTrainedModel):
    """ lstm 없는 lstm 모델 """
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config=config, add_pooling_layer=False)
        self.hidden_size = self.roberta.embeddings.word_embeddings.weight.data.shape[1]

        self.fc = Linear(self.hidden_size, self.hidden_size * 2)
        
        self.fc2 = Linear(self.hidden_size * 2, self.hidden_size)

        self.dense = Linear(self.hidden_size, config.num_labels)

        self.dropout = Dropout(0.2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]  # sequence = (batch, seq, hidden)
        
        output = self.dropout(self.fc(sequence_output)) # output = (batch, seq, hidden)
        
        output = self.dropout(self.fc2(output)) # output = (batch, seq, hidden)

        logits = self.dense(output) # logits = (batch, seq, 2)

        start_logits, end_logits = logits.split(
            1, dim=-1
        )  # start_logits, end_logits = (batch, seq)

        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)  # seq
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss, start_logits=start_logits, end_logits=end_logits,
        )
