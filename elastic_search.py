import json
import re
import kss
from tqdm import tqdm
import pandas as pd
from datasets import load_from_disk

from elasticsearch import Elasticsearch


def main():
    # load wikipeida documents
    path = "/opt/ml/data_v2/wikipeida_documents.json"
    with open(path, "r") as f:
        wiki = json.load(f)

    # wikipedia 문서 앞에 "!!{title}!!"를 삽입해 엘라스틱서치가 좀 더 잘 찾을 수 있도록 합니다
    # user_dict에 title을 추가하기 위해 title을 따로 뽑아줍니다
    titles = []
    for i in range(len(wiki)):
        context = wiki[f"{i}"]["text"]
        context = "!!" + wiki[f"{i}"]["title"] + "!!" + context
        wiki[f"{i}"]["text"] = context

        title = wiki[f"{i}"]["title"]
        title = re.sub('[()]', '', title).split()
        for t in title:
            titles.append(t.strip())

    titles = list(set(titles))
    if '' in titles:
        titles.pop(titles.index(''))

    # 엘라스틱 서치 설정
    INDEX_NAME = "wiki"
    index_config = {
        "settings": {
            "analysis": {
                "filter": {
                    "my_stop_filter": {
                        "type": "stop",
                        "stopwords_path": "my_stopwords.txt",  # /etc/elastic안에 txt파일이 존재해야 댑니다
                    }
                },
                "tokenizer": {
                    "my_nori_tokenizer": {
                        "type": "nori_tokenizer", # 노리 형태소 깔아야대는데 에러나면 맨위에 참고해서 깔기
                        "user_dictionary_rules": titles
                    }
                },
                "analyzer": {
                    "my_nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",  # 위에서 정의한 my_nori_tokenizer
                        "decompound_mode": "mixed",
                        "filter": ["my_stop_filter"],  # 위에서 정의한 stopword
                    }
                },
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "nori_analyzer"},
                "title": {"type": "text"},
                "document_id": {"type": "long"}
            },
        },
    }

    es = Elasticsearch("localhost:9200")

    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, body=index_config)

    # 엘라스틱 서치가 문서를 인덱싱합니다
    for doc_id, doc in tqdm(wiki.items(), total=len(wiki)):
        es.index(index=INDEX_NAME, id=doc_id, body=doc)

    path = "/opt/ml/data_v2/test_dataset"
    datasets = load_from_disk(path)
    print(datasets)

    query = datasets["validation"]["question"]
    id = datasets["validation"]["id"]

    total = []
    for i, q in enumerate(query):
        q = q.replace("~", "-")  # 쿼리에 ~가 있는 경우 에러가 나기 때문에 전처리한다
        res = es.search(index=INDEX_NAME, q=q, size=10)  # 10개의 문서를 반환합니다

        # id마다 조회된 10개의 문서를 'text{j}' column에 기록하고 'score{j}'에 각 문서에 대한 socre를 기록합니다
        tmp = {"id": id[i]}
        tmp.update(
            {f"text{j+1}": res["hits"]["hits"][j]["_source"]["text"] for j in range(10)}
        )
        tmp.update({f"score{j+1}": res["hits"]["hits"][j]["_score"] for j in range(10)})

        total.append(tmp)

    # 조회된 모든 문서를 DataFrame형태로 변환하고 csv파일로 저장합니다
    df = pd.DataFrame(total)
    df.to_csv("add_context_test_dataset.csv")


if __name__ == "__main__":
    main()
