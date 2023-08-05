#!/bin/bash
conda env create -f requirements/environment.yml
conda activate network-diffusion
pip install -r requirements/production.txt
pip install -r requirements/develop.txt
