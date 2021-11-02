import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, CrossEntropyLoss

from transformers import AutoConfig, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class lstmOnRoberta(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained("klue/roberta-large")
        self.num_labels = self.config.num_labels

        self.roberta = RobertaModel.from_pretrained(
            "klue/roberta-large", config=self.config
        )
        self.hidden_size = self.roberta.embeddings.word_embeddings.weight.data.shape[1]

        self.lstm1 = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )  # (batch, seq_length, 2*hidden_size)

        self.lstm2 = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = Linear(self.hidden_size * 4, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]  # sequence = (8, 384, 1024)

        start_pos_output, (hidden, cell) = self.lstm1(
            sequence_output
        )  # output = (8, 384, 2048), hidden = (4, 8, 1024), cell = (4, 8, 1024)

        end_pos_output, (hidden2, cell2) = self.lstm2(
            sequence_output, (hidden, cell)
        )  # output = (8, 384, 2048)

        output = torch.cat(
            (start_pos_output, end_pos_output), dim=2
        )  # output = (8, 384, 4096)

        logits = self.fc(output)  # logits = (8, 384, 2)

        start_logits, end_logits = logits.split(
            1, dim=-1
        )  # start_logits, end_logits = (8, 384)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)  # 384
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss, start_logits=start_logits, end_logits=end_logits,
        )

class lstmOnRoberta2(nn.Module):
    def __init__(self, training_args, dropout=0.1):
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
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )  # (batch, seq_length, 2*hidden_size)

        self.h0 = torch.randn(2, self.training_args.per_device_train_batch_size, self.hidden_size).cuda()
        self.c0 = torch.randn(2, self.training_args.per_device_train_batch_size, self.hidden_size).cuda()

        self.fc = Linear(self.hidden_size * 2, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]  # sequence = (8, 384, 1024)

        output, (hidden, cell) = self.lstm(sequence_output)

        logits = self.fc(output)  # logits = (8, 384, 2)

        start_logits, end_logits = logits.split(
            1, dim=-1
        )  # start_logits, end_logits = (8, 384)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)  # 384
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss, start_logits=start_logits, end_logits=end_logits,
        )