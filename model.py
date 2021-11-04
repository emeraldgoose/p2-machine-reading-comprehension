import torch
import torch.nn as nn
from torch.nn import functional as F
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


class ConvModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config=config, add_pooling_layer=False)
        self.hidden_size = self.roberta.embeddings.word_embeddings.weight.data.shape[1]

        self.conv1d_layer1 = nn.Conv1d(self.hidden_size, 1024, kernel_size=1)
        self.conv1d_layer3 = nn.Conv1d(self.hidden_size, 1024, kernel_size=3, padding=1)
        self.conv1d_layer5 = nn.Conv1d(self.hidden_size, 1024, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(0.5)

        self.dense = nn.Linear(1024 * 3, 2, bias=True)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
    ):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]  # Convolution 연산을 위해 Transpose (B * hidden_size * max_seq_legth)
        conv_input = sequence_output.transpose(1, 2)  # Conv 연산을 위한 Transpose (B * hidden_size * max_seq_length)
        conv_output1 = F.relu(self.conv1d_layer1(conv_input))  # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output3 = F.relu(self.conv1d_layer3(conv_input))  # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output5 = F.relu(self.conv1d_layer5(conv_input))  # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)

        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1)  # Concatenation (B * num_conv_filter x 3 * max_seq_legth)
        concat_output = concat_output.transpose(1, 2)
        concat_output = self.dropout(concat_output)

        logits = self.dense(concat_output)

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