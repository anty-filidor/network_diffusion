#!/bin/zsh

# pip deployment
python setup.py sdist bdist_wheel
echo 'anty-filidor' | twine upload  dist/*

# conda skeleton
mkdir conda_build
cd conda_build
conda skeleton pypi network_diffusion
conda-build network_diffusion --output-folder .

conda convert -f --platform all /usr/local/anaconda3/conda-bld/osx-64/network_diffusion-0.5.29-py37_0.tar.bz2
mkdir osx-64
cp /usr/local/anaconda3/conda-bld/osx-64/network_diffusion-0.5.29-py37_0.tar.bz2 osx-64/network_diffusion-0.5.29-py37_0.tar.bz2

# conda deployment
anaconda upload linux-64/network_diffusion-0.5.29-py37_0.tar.bz2
anaconda upload osx-64/network_diffusion-0.5.29-py37_0.tar.bz2
anaconda upload win-64/network_diffusion-0.5.29-py37_0.tar.bz2

# https://packaging.python.org/tutorials/packaging-projects/
# https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html
