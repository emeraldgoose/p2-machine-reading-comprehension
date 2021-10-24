# mrc-level2-nlp-10

-------------------------------

## 소개

MRC dataset version control

## 설치 방법

```
# dvc
pip install dvc
```



## 파일 구성

```
.dvc            # 데이터 저장된 폴더
data.dvc        # 데이터 분할 형식 정의
```



## data load & gdrive connection

```
git checkout dataset
dvc pull # 현재 폴더에 data 폴더 생성 (원하는 곳으로 옮겨주시면 됩니다) 
dvc remote add -d data gdrive://0AI9DszIpkCl9Uk9PVA
```



## version

```
tag 
v1.0
v1.1 - 초기 상태 데이터
```
