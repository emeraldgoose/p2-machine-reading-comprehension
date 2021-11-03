from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np

model_checkpoint = "klue/bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

import kss
import tqdm

train_path = "/opt/ml/code/train_dataset_.csv"
valid_path = "/opt/ml/code/valid_dataset_.csv"

train_data = pd.read_csv(train_path, index_col = 0)
valid_data = pd.read_csv(valid_path, index_col = 0)

class Sentence_finder_Dataset_maker():
    
    def __init__(self, data_path, tokenizer):
        self.data = pd.read_csv(data_path, index_col = 0)
        self.tokenizer = tokenizer
        self.results = []

    def _combine_sentences_one_record (self, idx, n= 3):
        query = self.data.iloc[idx].question
        answer = eval(self.data.iloc[idx].answers)
        context = self.data.iloc[idx].context
        document_id = self.data.iloc[idx].document_id

        # to update new start and end index in new sentences
        sentences = kss.split_sentences(context)

        ## find the sentence where has answer
        answer_idx = 0

        for i, sentence in enumerate(sentences):
            sentence_end = context.find(sentence) + len(sentence)

            if answer["answer_end"] < sentence_end:
                answer_idx = i
                sentence_end -= len(sentence)
                break

        new_answer_start = answer["answer_start"] - sentence_end

        results = []

        for i in range(len(sentences)-n+1):
            combined_sentences = "".join(sentences[i:i+n])

            if answer_idx >= i and answer_idx < i+n:
                chars_before = combined_sentences.find(sentences[answer_idx])
                real_answer_start = chars_before + new_answer_start
                real_answer_end = real_answer_start + len(answer["text"])
                find_answer = 1

            else:
                real_answer_start = -1
                real_answer_end = -1
                find_answer = 0

            new_document_id = str(document_id) +"_"+ str(i)

            result = [combined_sentences, query, 
            {"answer_start":real_answer_start, "text":answer["text"], "answer_end":real_answer_end},
             new_document_id, find_answer]
             
            results.append(result)

        return results

    def combine_sentences(self, n = 3, save_path = None):

        if len(self.results) != 0:
            self.results = []

        for i in tqdm.tqdm(range(len(self.data))):
            results = self._combine_sentences_one_record(i)
            self.results += results

        if save_path is not None:
            pd.DataFrame(self.results, columns = ["context, query, answer, document_id, answerable"]).to_csv(save_path)

s_finder = Sentence_finder_Dataset_maker(valid_path, tokenizer)
s_finder.combine_sentences(save_path = "valid_dataset_for_classifier.csv")

s_finder = Sentence_finder_Dataset_maker(train_path, tokenizer)
s_finder.combine_sentences(save_path = "train_dataset_for_classifier.csv")


