# Network Diffusion - spreading models in complex networks

[![License: GPL](https://img.shields.io/github/license/anty-filidor/network_diffusion)](https://www.gnu.org/licenses/gpl-3.0.html)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4592269.svg)](https://doi.org/10.5281/zenodo.4592269)

[![PyPI version](https://badge.fury.io/py/network-diffusion.svg)](https://badge.fury.io/py/network-diffusion)

![Tests](https://github.com/anty-filidor/network_diffusion/actions/workflows/tests.yml/badge.svg)
![Builds](https://github.com/anty-filidor/network_diffusion/actions/workflows/package-build.yml/badge.svg)
[![Docs](https://readthedocs.org/projects/network-diffusion/badge/?version=latest)](https://network-diffusion.readthedocs.io/en/latest)
[![codecov](https://codecov.io/gh/anty-filidor/network_diffusion/branch/package-simplification/graph/badge.svg?token=LF52GAD73F)](https://codecov.io/gh/anty-filidor/network_diffusion)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fanty-filidor%2Fnetwork_diffusion.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fanty-filidor%2Fnetwork_diffusion?ref=badge_shield)

This Python library provides a versatile toolkit for simulating diffusion
processes in complex networks. It offers support for various types of models,
including temporal models, multilayer models, and combinations of both.

<p align="center"> <b>Documentation</b> is available <a href="https://network-diffusion.readthedocs.io/en/latest/">here</a>! </p>

## Key Features

- **Complex Network Simulation**: The library enables users to simulate
  diffusion processes in complex networks with ease. Whether you are studying
  information spread, disease propagation, or any other diffusion phenomena,
  this library has you covered.

- **Temporal Models**: You can work with temporal models, allowing you to
  capture the dynamics of processes over time. These temporal models can be
  created using regular time windows or leverage
  [CogSnet](https://www.researchgate.net/publication/348341904_Social_Networks_through_the_Prism_of_Cognition).

- **Multilayer Networks**: The library supports multilayer networks, which are
  essential for modeling real-world systems with interconnected layers of
  complexity.

- **Predefined Models**: You have the option to use predefined diffusion models
  such as the Linear Threshold Model, Independent Cascade Model, and more.
  These models simplify the simulation process, allowing you to focus on your
  specific research questions.

- **Custom Models**: Additionally, Nwtwork Diffusion allows you to define your
  own diffusion models using open interfaces, providing flexibility for
  researchers to tailor simulations to their unique requirements.

- **Centrality Measures**: The library provides a wide range of centrality
  measures specifically designed for multilayer networks. These measures can be
  valuable for selecting influential seed nodes in diffusion processes.

- **NetworkX Compatibility**: Last but not least, the package is built on top
  of NetworkX, ensuring seamless compatibility with this popular Python library
  for network analysis. You can easily integrate it into your existing
  NetworkX-based workflows.

## Installation

**To install package run this command: `pip install network_diffusion`**.
Please note, that currently we support Linux and MacOS only.

If you like the package, please cite us as:

```
@INPROCEEDINGS{czuba2022networkdiffusion,
    author={Czuba, Micha\l{} and Br\'{o}dka, Piotr},
    booktitle={2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA)},
    title={Simulating Spreading of Multiple Interacting Processes in Complex Networks},
    year={2022},
    month={oct},
    volume={},
    number={},
    pages={1-10},
    publisher={IEEE},
    address={Shenzhen, China},
    doi={10.1109/DSAA54385.2022.10032425},
}
```

## New features incoming

A board with issues and state of the progress torwards implementing new
functionalities can be found
[here](https://github.com/users/anty-filidor/projects/6/views/1).

## About us

This library is developed and maintained by Network Science Lab at
[WUST](https://networks.pwr.edu.pl/) and external partners. For more
information and updates, please visit our :
[website](https://networks.pwr.edu.pl/).
