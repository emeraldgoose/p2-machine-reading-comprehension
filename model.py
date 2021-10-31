import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, CrossEntropyLoss, Dropout, GELU

from transformers import AutoConfig, RobertaModel, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class lstmOnRoberta(nn.Module):
    def __init__(self, training_args, dropout=0.2):
        super().__init__()
        self.training_args = training_args
        self.config = AutoConfig.from_pretrained("klue/roberta-large")
        self.num_labels = self.config.num_labels
        self.roberta = RobertaModel.from_pretrained(
            "klue/roberta-large", config=self.config
        )
        self.hidden_size = self.roberta.embeddings.word_embeddings.weight.data.shape[1]

        self.lstm = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )  # (batch, seq_length, 2*hidden_size)

        self.fc = Linear(self.hidden_size * 2, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]  # sequence = (batch, seq, hidden)

        output, (hidden, cell) = self.lstm(sequence_output)

        logits = self.fc(output)  # logits = (batch, seq, 2)

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

class mlpOnRoberta(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config=config, add_pooling_layer=False)
        
        self.hidden_size = self.roberta.embeddings.word_embeddings.weight.data.shape[1]
        
        self.fc1 = Linear(self.hidden_size, self.hidden_size)
        self.fc2 = Linear(self.hidden_size, self.hidden_size)
        self.fc3 = Linear(self.hidden_size, self.hidden_size)
        self.fc4 = Linear(self.hidden_size, self.num_labels)
        
        self.dropout = Dropout(0.5)
        self.gelu = GELU()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]  # sequence = (batch, seq, hidden)

        output = self.fc1(sequence_output)
        
        output = self.dropout(self.gelu(output))

        output = self.fc2(output)

        output = self.dropout(self.gelu(output))

        output = self.fc3(output)

        output = self.dropout(self.gelu(output))

        logits = self.fc4(output)

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
