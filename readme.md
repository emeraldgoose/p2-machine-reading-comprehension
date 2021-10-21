# mrc-level2-nlp-10

-------------------------------

## 소개

MRC dataset version control

## 설치 방법

### 요구 사항

```
# dvc
pip install dvc
```

## 파일 구성


### 저장소 구조

```
  .dvc            # 전체 데이터
  data.dvc        # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
```

## 데이터 소개

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![데이터 분포](./assets/dataset.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```python
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 
