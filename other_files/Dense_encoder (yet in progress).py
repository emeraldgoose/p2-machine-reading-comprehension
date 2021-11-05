"""
아직 기안중이라 미완성입니다.

Dense Encoder는 Encoding을 위한 Dataset을 먼저 생성하고, 해당 Dataset을 바탕으로 학습이 진행됩니다.
밑의 "Dense_dataset_maker"는 Dense Encoding을 위한 Dataset을 생성하는 클래스입니다.

후에 해당 Dataset을 바탕으로 학습하는 Encoder도 추가로 넣도록 하겠습니다.

이상입니다.
"""

import os
import json
import time
import faiss
import pickle
import numpy as np
import random
import pandas as pd

import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

from transformers import AutoTokenizer
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

class Dense_dataset_maker:

    def __init__(
        self, tokenizer_model,
        train_path = "/opt/ml/code/train_dataset.csv", valid_path = "/opt/ml/code/valid_dataset.csv",
        n_neg_samples = 5):

        """
        tokenize_fn : 토큰화를 진행할 모델
        train_path : train_dataset의 경로
        valid_path : valid_dataset의 경로
        n_neg_samples : negative sampling 할 때 negative sample의 개수
        """

        self.n_neg_samples = n_neg_samples

        tr_df = pd.read_csv(train_path, index_col = 0)
        val_df = pd.read_csv(valid_path, index_col = 0)

        self.total_df = pd.concat([tr_df, val_df])
        self.total_df.index = list(range(len(self.total_df)))
        
        id_and_index = zip(list(range(len(self.total_df))), self.total_df.document_id)
        self.id_to_index = {doc_id : index for (index, doc_id) in id_and_index}        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        #self.tokenizer = tokenizer_model

    def _get_negative_samples (self):
        """
        
        negative sampling을 위한 class
        - [positive sample, negative sample * n_neg] 의 형태로 document list을 반환함. 
        """

        context_with_negative = []

        for positive_sample_id in tqdm.tqdm(self.total_df.document_id.values):

            positive_with_negative = [positive_sample_id]

            # negative sample index를 추출 (중복 없이 n_neg_sample개만큼 추출)

            while True:

                random_negative_samples = np.random.choice(self.total_df.document_id.values, self.n_neg_samples)

                if positive_sample_id in random_negative_samples:
                    continue

                else:
                    positive_with_negative += random_negative_samples.tolist()

                if len(positive_with_negative) == self.n_neg_samples +1:
                    break

            # index를 context로 전환

            context_list = []

            for document_id in positive_with_negative:
                loc = self.id_to_index[document_id]
                context = self.total_df.iloc[loc].context
                context_list.append(context)

            context_with_negative += context_list

        return context_with_negative

    def run(self, save_path = None):
      
        """
        앞서 negative sampling한 데이터를 tokenize한 후에, dataset 형태로 반환
        만약 save_path = {save_path}를 설정하면, 해당 데이터셋이 pytorch 형태로 {save_path}에 저장됨.
        """

        # get negative samples and tokenize
        # cont : [25152, 512], query : [4192, 512]

        context_with_negative= self._get_negative_samples()
        context_sequence = self.tokenizer(context_with_negative, padding = "max_length", truncation = True, return_tensors = "pt")
        query_sequence = self.tokenizer(self.total_df.question.values.tolist(), padding = "max_length", truncation = True, return_tensors = "pt")
        
        max_len = context_sequence["input_ids"].shape[-1] # 512

        #원래 (n_document * n_neg +1, max_len)이지만,
        #[pos, n_neg+1] 쌍으로 구분해주기 위하여 [n_document, n_neg+1, max_len]로 수정

        for k,_ in context_sequence.items():
            context_sequence[k] = context_sequence[k].reshape(-1, self.n_neg_samples+1, max_len)

        train_dataset = TensorDataset(context_sequence['input_ids'], context_sequence['attention_mask'], context_sequence['token_type_ids'], 
                        query_sequence['input_ids'], query_sequence['attention_mask'], query_sequence['token_type_ids'])

        if save_path is not None:
            torch.save(train_dataset, save_path)

        return train_dataset

retriever = Dense_dataset_maker("klue/roberta-base")
dataset = retriever.run("dense_encoder_dataset.pt")
