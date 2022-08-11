#!/bin/bash

# create a conda env for development

conda env create -f environment.yml
pip install -r production.txt
pip install -r develop.txt
