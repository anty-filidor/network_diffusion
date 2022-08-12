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


add_tag:
  stage: publish
  before_script:
    - git config --global user.name "${GITLAB_USER_NAME}"
    - git config --global user.email "${GITLAB_USER_EMAIL}"
    - git remote rm origin
    - git remote add origin
      https://oauth2:${AUTH_TOKEN}@gitlab.com/${CI_PROJECT_PATH}
  script:
    - VER=$(cat nn_inference/__init__.py | grep __version__ | cut -d'"' -f2)
    - git tag -a ${VER} -m "Version ${VER}"
    - git push origin --tags
  rules:
    - if: $CI_COMMIT_BRANCH == "master"
