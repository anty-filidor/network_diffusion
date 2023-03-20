#!/bin/bash

# create a conda env for development
conda env create -f environment.yml

pip install -r production.txt
pip install -r develop.txt

# compile sphinx
# python -m sphinx -T -E -b html -d docs/doctrees -D language=en docs/source docs/html
