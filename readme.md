# mrc-level2-nlp-10

## ODQA란?

**ODQA(Open Domain Question and Answering)**:

지문이 따로 주어지지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾은 후, 해당 문서에서 질문에 대한 대답을 찾는 문제

## 구조

본 조에서는 **Retriever-Reader Model**을 통해 ODQA 문제를 해결하고자 하였다.

- **Retriever** : Query가 주어졌을 때, Query에 대한 대답을 할 수 있는 문서를 문서 사전에서 찾아오는 모델
- **Reader** : Retriever를 통해 반환된 문서에서, Query와 가장 관련 깊은 Phrase를 찾아오는 모델


### 요구 사항

```bash
datasets==1.5.0
transformers==4.5.0
tqdm==4.41.1
pandas==1.1.4
scikit-learn==0.24.1
konlpy==0.5.2
Elasticsearch==7.15.1
```

### Elastic Search 및 Nori Analyzer 설치
```
apt-get update && apt-get install -y gnupg2
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee /etc/apt/sources.list.d/elastic-7.x.list
apt-get update && apt-get install elasticsearch
service elasticsearch start
cd /usr/share/elasticsearch
bin/elasticsearch-plugin install analysis-nori
service elasticsearch restart
```
## 설치 방법
```
git clone https://github.com/boostcampaitech2/mrc-level2-nlp-10.git

cd mrc-level2-nlp-10.git

bash ./install/install_requirements.sh
```


## 파일 구성


### 저장소 구조

```bash
./install/               # 요구사항 설치 파일 
./data_preprocessing/.   # 데이터 전처리
arguments.py             # 모델, 데이터셋 argument가 dataclass 의 형태로 저장
trainer_qa.py            # MRC 모델 평가에 대한 trainer 제공
utils_qa.py              # 기타 유틸 함수 제공 
preprocessing.py         # 학습을 위한 전처리 함수를 반환
postprocessing.py        # eval을 위한 후처리 함수를 반환
make_dataset.py          # 전처리가 적용된 데이터셋을 반환
modelList.py             # config, tokenizer, model를 반환
model.py                 # QA task를 수행하는 모델들
train.py                 # MRC 모델 학습 및 평가 
inference.py             # ODQA 모델 평가 또는 제출 파일 생성
elastic_search.py        # 엘라스틱서치를 통해 쿼리에 대한 문서를 조회
ensemble.py              # 여러 모델의 결과를 앙상블
sweep.yaml               # wandb sweep 설정 파일
```

## 데이터 소개

MRC 데이터의 경우, HuggingFace에서 제공하는 `datasets` 라이브러리를 이용하여 접근이 가능합니다. 해당 directory를 dataset_name 으로 저장한 후, 아래의 코드를 활용하여 불러올 수 있습니다.

```python=
# train_dataset을 불러오고 싶은 경우

from datasets import load_from_disk
dataset = load_from_disk("./data/train_dataset/")

train_dataset = dataset["train"]
valid_dataset = dataset["validation"]
```
Retrieval 과정에서 사용하는 문서 집합(corpus)은 `./data/wikipedia_documents.json` 경로로 저장되어있습니다. 약 5만 7천개의 unique 한 문서로 이루어져 있습니다.

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 `datasets`를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 ./data 구조입니다.

```python
# 전체 데이터
./data/
    # 학습에 사용할 데이터셋. train 과 validation 으로 구성
    ./train_dataset/
    # 제출에 사용될 데이터셋. validation 으로 구성
    ./test_dataset/
    # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
    ./wikipedia_documents.json
```

## 데이터 예시

### 실제 데이터셋의 형태

- `title` : 문서(context)의 제목
- `context` : 문서의 내용
- `question` : 문서를 통해 답을 찾을 수 있는 질문
- `id` : question의 고유 id
- `answers` : 문서를 통해 도출해낼 수 있는 답
- `document_id` : 문서의 고유 id


## 훈련, 평가, 추론

### model
roberta-large, bert-base, electra-base위에 MLP를 쌓아 QA 리더 모델을 구성했습니다.  

각 레이어에서 나온 값들은 dropout(0.2)을 통과시키게 한 후 start logits, end logits를 반환하게 하여 주어진 context에서 답을 찾도록 했습니다.  

roberta와 conv1d를 사용해 인접한 토큰간의 관계를 볼 수 있도록 시도한 모델도 있습니다.  

각 사용한 pretrained 모델은 klue/roberta-large, klue/bert-base, monologg/koelectra-base-v3-discriminator이고 각 QA 모델의 구조는 `model.py`에서 볼 수 있습니다.


### train

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요. 

또한, wandb sweep이 세팅되어 있어 `sweep.yaml`와 아래 wandb 정보를 수정하시면 fine tuning이 가능하도록 작성되었습니다. 

wandb를 disabled하고 `defaults`를 수정하면 수동으로 하이퍼파라미터를 변경할 수 있습니다.

각 모델에 대한 weight와 eval 결과는 ./models/train_dataset/{model_name}에 저장됩니다.


```python
if __name__ == "__main__":
    defaults = dict(learning_rate=1e-5, epochs=2, weight_decay=0.009)

    wandb.init(
        config=defaults, project="", entity="", name="",
    )

    wandb.agent(entity="", project="", sweep_id="")

    config = wandb.config

    main(config)
```

```
# 학습 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --do_train --overwrite_output_dir
```

### eval

모델이 train하면서 `training_args.eval_step` 마다 evaluation을 진행합니다. 이때 Exact Match 스코어가 가장 높은 모델이 저장되도록 작성되어 따로 evaluation을 위한 코드를 실행할 필요가 없습니다.


### inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

* 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 `--model_name_or_path`에 ./model/train_dataset/{model_name}을 넣고 추론(`--do_predict`)을 진행하면 됩니다. 

* 추론 후 prediction_{model_name}.json와 nbest_prediction_{model_name}.json이 생성됩니다.

```
# ODQA 실행 (test_dataset 사용)
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/{model_name} --do_predict
```

### Ensemble
RobertaQA, BertQA, ElectraQA의 결과를 앙상블하는 `ensemble.py`가 작성되었습니다. 
```
python ensemble.py
```

### How to submit

`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--output_dir` 위치에 `predictions.json` 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

### 모델 학습 결과

다음은 실험한 MRC 모델의 결과를 보여줍니다.

|model|eval|Public LB|Private LB|
|-|-|-|-|
|RobertaForQuestionAnswering|72|53.75|50.56|
|RobertaQA|**74.58**|**60.00**|**60.56**|
|BertQA|57.08|.|.|
|ElectraQA|57.08|.|.|
|Roberta+Conv-1d|.|59.17|58.06|
|Ensemble(RobertaQA + BertQA + ElectraQA)|.|58.75|58.61|
