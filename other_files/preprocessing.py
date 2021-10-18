
"""
from datasets import load_from_disk

dataset = load_from_disk("/opt/ml/data/train_dataset")

dataset["train"].to_csv("train_dataset.csv")
dataset["validation"].to_csv("valid_dataset.csv")
"""

# Code above converts dataset files to csv files. if you don't have the datasets in csv file, please remove the comments and run the codes
# 위 코드는 Huggingface에서 제공하는 dataset 파일을 csv 파일로 전환하는 코드입니다. 만약 csv 파일이 없다면 위 코드도 실행해주세요.

# import library
import pandas as pd
import numpy as np
import re
import tqdm

csv_path = "/opt/ml/code/train_dataset.csv" # path to train datasets in csv file

class Preprocessor():
  
    """
    __init__: 
      "Preprocessor" 클래스의 생성자입니다. "csv_path"을 입력으로 받습니다.
      
    _answers_to_dictionary:
      csv file은 모든 파일을 "str" type으로 저장합니다. "answers" attribute는 dict 파일로 구성되어 있기 때문에,
      string type을 dict type으로 바꿔주는 역할을 수행합니다.
      
      answers : string type으로 변환된 dictionary type
      
    _replace_and_get_offset:
      "re.sub"와 동일한 역할을 수행합니다. 다만, 글자가 다른 글자로 대체되는 만큼 그에 따른 answers["text"]의 index 변화량 또한 반환합니다.
      다만 이 코드에서는 대체되는 글자와 대체하는 글자의 크기가 같기 때문에 별 의미는 없다고 생각합니다.
      
    _run_one_record:
      하나의 record를 전처리합니다. "context", "question", "answer", "document_id" 외의 정보는 불필요하다 판단하여 계산에서 제외합니다.
      
    run:
      데이터 셋 내의 모든 record를 전처리합니다.
      만약 save_path에 저장경로를 입력할 시, 전처리된 파일을 해당 경로에 csv 파일로 저장합니다.
      저장 경로를 입력하지 않을 시, {instance_name}.new_csv을 통해 전처리된 데이터를 불러올 수 있습니다.
    """
 
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, index_col = 0)
        self.new_csv = []

    def _answers_to_dictionary(self, answers):

        def array(obje, dtype = object):
            return obje.pop()

        return eval(answers)

    def _replace_and_get_offset(self, pattern, rpl, context, standard = None):

        offset = 0

        re_iter = re.finditer(pattern, context)

        if standard is not None: 
            for i in re_iter:
                span = i.span()
                if span[0] < standard:
                    offset += (len(rpl)-(span[1]-span[0]))
                else:
                    break
        
        context = re.sub(pattern, rpl, context)

        if standard is not None:
            return standard + offset, context
        else:
            return context

    def _run_one_record(self, idx):

        context = self.df.iloc[idx].values[1]
        question = self.df.iloc[idx].values[2]
        answers = self._answers_to_dictionary(self.df.iloc[idx].values[4])
        answer = answers["text"]
        document_id = self.df.iloc[idx].values[5]

        # replace context
        start_idx = answers["answer_start"]

        new_standard, context = self._replace_and_get_offset("[“”‘’\"\']", "\'", context, start_idx)
        new_standard, context = self._replace_and_get_offset("[〈<＜「≪《『]", "<", context, new_standard)
        new_standard, context = self._replace_and_get_offset("[〉>＞」≫》』]", ">", context, new_standard)
        new_standard, context = self._replace_and_get_offset("\\\\n|\\n| {2, }", " ", context, new_standard)

        #replace word

        answer = self._replace_and_get_offset("[“”‘’\"\']", "\'", answer)
        answer = self._replace_and_get_offset("[〈<＜「≪《『]", "<", answer)
        answer = self._replace_and_get_offset("[〉>＞」≫》』]", ">", answer)
        answer  = self._replace_and_get_offset("\\\\n|\\n| {2, }", " ", answer)

        answers["answer_start"] = new_standard
        answers["text"] = answer
        answers["answer_end"] = new_standard + len(answer)

        return [context, question, answers, document_id]

    def run(self, save_path = None):

        self.new_csv = []
        
        for i in tqdm.tqdm(range(len(self.df))):
            self.new_csv.append(self._run_one_record(i))
        
        print("preprocessing done!")

        if save_path is not None:
            pd.DataFrame(self.new_csv, columns = ["context", "question", "answers","document_id"]).to_csv(save_path)

# how to run the code
"""
preprocessor = Preprocessor(csv_path)
preprocessor.run("after_preprocessing_train.csv")
"""

# Json_file preprocessor
# wikipedia_documents.json 파일을 전처리하기 위한 클래스입니다.

import json

json_path = "/opt/ml/data/wikipedia_documents.json"

class Preprocessor_json():

    def __init__(self, json_path):

        self.wiki = json.load(open(json_path, "r", encoding="utf-8"))
        self.contexts = list(dict.fromkeys([v["text"] for v in self.wiki.values()]))

    def _replace_one_record(self, idx):

        context = re.sub("[“”‘’\"\']", "\'", self.contexts[idx])
        context = re.sub("[〈<＜「≪《『]", "<", context)
        context = re.sub("[〉>＞」≫》』]", ">", context)
        context = re.sub("\\\\n|\\n| {2, }", " ", context)

        #print(f"original :\n{self.contexts[idx]}")
        #print(f"\n\nchanged :\n{context}")

        return context

    def run(self, save_path = None):
        
        for i in tqdm.tqdm(range(len(self.contexts))):
            self.wiki[str(i)]["text"] = self._replace_one_record(i)
        
        print("preprocessing done!")

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(self.wiki, f) 

# how to run
"""
j_processor = Preprocessor_json(json_path)
j_processor.run("processed_wikipedia.json")
"""
