# mrc-level2-nlp-10

-------------------------------

## 소개

MRC dataset version control

## 설치 방법

```
wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
apt-get update
apt-get install dvc
```



## 파일 구성

```
.dvc            # 데이터 저장된 폴더
data.dvc        # 데이터 분할 저장
```



## data load & gdrive connection

```
cd /opt/ml
git init
git remote add origin https://github.com/boostcampaitech2/mrc-level2-nlp-10.git
git pull
git checkout dataset
rm -rf data
dvc pull
```



## version

```
tag 
v1.0 - 초기 상태 데이터
v2.0 - #, \n\n, \n 문자 제거
v3.0 - “”‘’ -> \', 
       〈<＜「≪《『 -> <,
       〉>＞」≫》』 -> >
```

## version control

```
# 베포된 버전 목록 확인
git tag

# ex) v1.0으로 이동 
git checkout v1.0
```
