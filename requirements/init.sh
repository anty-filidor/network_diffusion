#!/bin/bash

conda env create -f environment.yml
conda activate network-diffusion
pip install -r production.txt
pip install -r develop.txt
