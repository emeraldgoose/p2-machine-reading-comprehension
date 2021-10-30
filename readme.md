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
chmod +x load_dataset.sh
./load_dataset.sh
구글 인증해주시면 데이터 생싱이 
```



## version

```
tag 
v1.0
v1.1 - 초기 상태 데이터
v2.0
v2.1 - #, \n\n, \n 문자 제거
v3.0 - “”‘’ -> \', 
       〈<＜「≪《『 -> <,
       〉>＞」≫》』 -> >
```
