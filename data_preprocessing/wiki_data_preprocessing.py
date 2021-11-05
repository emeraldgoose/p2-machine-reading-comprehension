import json
import re

json_path = "/opt/ml/data/wikipedia_documents.json"

with open(json_path, 'r') as f:
    json_data = json.load(f)

for i in json_data:
    example = json_data[i]

    print(example)

    example['text'] = re.sub(r'\n', " ", example['text'])
    example['text'] = re.sub(r"\\n", " ", example['text'])
    example['text'] = re.sub(r'#', " ", example['text'])

    example['text'] = re.sub(r"[“”‘’]", "\'", example['text'])
    example['text'] = re.sub(r"[〈<＜「≪《『]", "<", example['text'])
    example['text'] = re.sub(r"[〉>＞」≫》』]", ">", example['text'])

    example['text'] = re.sub(r"\s+", " ", example['text'])
    example['text'] = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "",
                             example['text'])

    del example['corpus_source']
    del example['url']
    del example['domain']
    del example['author']
    del example['html']
    del example['document_id']

    print(example)

    json_data[i] = example

with open("/opt/ml/data/wikipedia_documents_v3_no_remove.json", 'w') as outfile:
    json.dump(json_data, outfile)