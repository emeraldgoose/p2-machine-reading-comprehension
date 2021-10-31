import pandas as pd

train_data = "/opt/ml/code/train_dataset_no_tilde.csv"
valid_data = "/opt/ml/code/valid_dataset_no_tilde.csv"

# train data와 valid 데이터를 하나로 묶어 dictionary를 생성하는 함수

def convert(train_path, valid_path):
    # make a list of dictionary
    total_data = [] # {context, question, document_id}

    train_data = pd.read_csv(train_path, index_col = 0)
    valid_data = pd.read_csv(valid_path, index_col = 0)

    total_df = pd.concat([train_df, valid_df])

    for i in range(len(total_df)):
        temp_data = total_df.iloc[i]
        total_data.append({"text":temp_data.context, "question":temp_data.question, "document_id": temp_data.document_id})

    return total_data

total_data = convert(train_data, valid_data)   

# retriever 결과를 반환하는 함수

import tqdm
import re

def show_the_result(retriever, total_data):

    results = [0,0,0,0,0] # [top1, top5, top10, top20, not_found] 

    total_data_len = len(total_data)
    
    for ind in tqdm.tqdm(range(total_data_len)):

        temp_data = total_data[ind]
        query = re.sub("~","-", temp_data["question"])
        query = re.sub("/","", query)
        document_id = temp_data["document_id"]

        hit_ones = retriever.search(query, 20)["hits"]["hits"]

        if hit_ones: #만약 검출이 되었다면
            result = [hit_one["_source"]["document_id"] for hit_one in hit_ones]
            
            if document_id in result:
                found_index = result.index(document_id)
                if found_index == 0:
                    results[0] += 1
                elif found_index <= 5:
                    results[1] += 1
                elif found_index <= 10:
                    results[2] += 1
                else:
                    results[3] += 1

            else:
                results[4] += 1

    return results

results = show_the_result(retriever, total_data)

# 결과를 보기 좋게 표현해주는 함수

def pretty_result(result):
    total = sum(result)
    top_1 = result[0]
    top_5 = sum(result[:2])
    top_10 = sum(result[:3])
    top_20 = sum(result[:4])

    print(f"===Retrieval Result===\n")
    print(f"top 1 : {top_1*100/total}%")
    print(f"top 5 : {top_5*100/total}%")
    print(f"top 10 : {top_10*100/total}%")
    print(f"top 20 : {top_20*100/total}%")
    print(f"failed to predict : {result[-1]*100/total}%")

pretty_result(results)
