#!/bin/bash

cd /opt/ml
git init && git remote add origin https://github.com/boostcampaitech2/mrc-level2-nlp-10.git
git pull
git checkout dataset
dvc pull
