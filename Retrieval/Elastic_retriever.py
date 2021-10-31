# import library

from elasticsearch import Elasticsearch
import json
import tqdm

class Elastic_retriever():

    def __init__(self, wiki_json_path):
        self.wiki_list = [] # saves text and document id only
        
        # load wikipedia data and drop needless informations 
        with open(wiki_json_path, "r", encoding = "utf-8") as f:
            wiki = json.load(f)

            for ind in range(len(wiki)):
                temp_wiki = wiki[str(ind)]
                self.wiki_list.append({"text": temp_wiki["text"], "document_id" : temp_wiki["document_id"]})

        del wiki # for memory usage
        self.es = Elasticsearch("localhost:9200")

    def _create_indice(self, index_name = None, index_config = None):

        if index_config is None:
            index_config = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "standard_analyzer": {
                                "type": "standard"
                            }
                        }
                    }
                },
                "mappings": {
                    "dynamic": "strict", 
                    "properties": {
                        "document_id": {"type": "long",},
                        "text": {"type": "text", "analyzer": "standard_analyzer"}
                        }
                    }
                }

        if index_name is None:
            index_name = 'klue_mrc_wikipedia_index'

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)

        self.es.indices.create(index=index_name, body=index_config, ignore=400)

    def _populate_index(self, index_name = None):
        
        if index_name is None:
            index_name = 'klue_mrc_wikipedia_index'

        for i in tqdm.tqdm(range(len(self.wiki_list))):
            index_status = self.es.index(index = index_name, id = i, body = self.wiki_list[i])
        
        n_records = es.count(index = index_name)["count"]
        print(f'Succesfully loaded {n_records} into {index_name}')

    def config_and_index(self, index_name = None, index_config = None):
        self._create_indice(index_name, index_config)
        self._populate_index(index_name)

    def search(self, query, num_return, index_name = None):
        if index_name is None:
            index_name = 'klue_mrc_wikipedia_index'
        answer = self.es.search(index=index_name, q = query, size = num_return)
        return answer
        
# How to run
 
wiki_json_path = "/opt/ml/code/preprocessed_json_v3.json"

retriever = Elastic_retriever(wiki_json_path)
retriever._create_indice()
retriever._populate_index()

# How to Search

retriever.search("이순신 장군은 어느 전쟁에서 사망하였는가", 3)
