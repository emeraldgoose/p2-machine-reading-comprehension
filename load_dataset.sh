#!/bin/bash

mkdir dvc
cd dvc
git init
git remote add origin https://github.com/boostcampaitech2/mrc-level2-nlp-10.git
git pull origin dataset
dvc pull
