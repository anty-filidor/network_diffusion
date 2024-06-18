# Network Diffusion - spreading models in networks

[![Licence](https://img.shields.io/github/license/anty-filidor/network_diffusion)](https://opensource.org/license/mit)
[![PyPI version](https://badge.fury.io/py/network-diffusion.svg)](https://badge.fury.io/py/network-diffusion)

![Tests](https://github.com/anty-filidor/network_diffusion/actions/workflows/tests.yml/badge.svg)
![Builds](https://github.com/anty-filidor/network_diffusion/actions/workflows/package-build.yml/badge.svg)
[![Docs](https://readthedocs.org/projects/network-diffusion/badge/?version=latest)](https://network-diffusion.readthedocs.io/en/latest)
[![codecov](https://codecov.io/gh/anty-filidor/network_diffusion/branch/package-simplification/graph/badge.svg?token=LF52GAD73F)](https://codecov.io/gh/anty-filidor/network_diffusion)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fanty-filidor%2Fnetwork_diffusion.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fanty-filidor%2Fnetwork_diffusion?ref=badge_shield)

This Python library provides a versatile toolkit for simulating diffusion
processes in complex networks. It offers support for various types of models,
including temporal models, multilayer models, and combinations of both.

## A Short Example

```python
import network_diffusion as nd

# define the model with its internal parameters
spreading_model = nd.models.MICModel(
    seeding_budget=[90, 10, 0],  # 95% act suspected, 10% infected, 0% recovered
    seed_selector=nd.seeding.RandomSeedSelector(),  # pick infected act randomly
    protocol="OR",  # how to aggregate impulses from the network's layers
    probability=0.5,  # probability of infection
)

# get the graph - a medium for spreading
network = nd.mln.functions.get_toy_network_piotr()

# perform the simulation that lasts four epochs
simulator = nd.Simulator(model=spreading_model, network=network)
logs = simulator.perform_propagation(n_epochs=3)

# obtain detailed logs for each actor in the form of JSON
raw_logs_json = logs.get_detailed_logs()

# or obtain aggregated logs for each of the network's layer
aggregated_logs_json = logs.get_aggragated_logs()

# or just save a summary of the experiment with all the experiment's details
logs.report(visualisation=True, path="my_experiment")
```

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
  essential for modelling real-world systems with interconnected layers of
  complexity.

- **Predefined Models**: You have the option to use predefined diffusion models
  such as the Linear Threshold Model, Independent Cascade Model, and more.
  These models simplify the simulation process, allowing you to focus on your
  specific research questions.

- **Custom Models**: Additionally, Network Diffusion allows you to define your
  own diffusion models using open interfaces, providing flexibility for
  researchers to tailor simulations to their unique requirements.

- **Centrality Measures**: The library provides a wide range of centrality
  measures specifically designed for multilayer networks. These measures can be
  valuable for selecting influential seed nodes in diffusion processes.

- **NetworkX Compatible**: Last but not least, the package is built on top of
  NetworkX, ensuring seamless compatibility with this popular Python library
  for network analysis. You can easily integrate it into your existing
  NetworkX-based workflows.

## Package installation

To install the package, run this command: `pip install network_diffusion`.
Please note that we currently support Linux, MacOS, and Windows, but the
package is mostly tested and developed on Unix-based systems.

To contribute, please clone the repo, switch to a new feature branch, and
install the environment:

```bash
conda env create -f env/conda.yml
conda activate network-diffusion
pip install -e .
```

## Documentation

<p align="center"> <b>Reference guide</b> is available <a href="https://network-diffusion.readthedocs.io/en/latest/">here</a>! </p>

Please bear in mind that **this project is still in development**, so the API
usually differs between versions. Nonetheless, the code is documented well, so
we encourage users to explore the repository. Another way to familiarise
yourself with the operating principles of `network_diffusion` are projects
which utilise it:

- Generator of a dataset with actors' spreading potentials - _v0.14.4_ -
  [repo](https://github.com/network-science-lab/infmax-simulator-icm-mln)
- Influence max. under LTM in multilayer networks - _v0.14.0 pre-release_ -
  [repo](https://github.com/anty-filidor/rank-refined-seeding-bc-infmax-mlnets-ltm)
- Comparison of spreading in various temporal network models - _v0.13.0_ -
  [repo](https://github.com/anty-filidor/bdma-experiments)
- Seed selection methods for ICM in multilayer networks - _v0.10.0_ -
  [repo](https://github.com/damian4060/Independent_Cascade_Model)
- Modelling coexisting spreading phenomena - _v0.6_ -
  [repo](https://github.com/anty-filidor/network_diffusion_examples)

## Citing us

If you used the package, please consider citing us:

```bibtex
@article{czuba2024networkdiffusion,
  title={Network Diffusion – Framework to Simulate Spreading Processes in Complex Networks},
  author={
      Czuba, Micha{\l} and Nurek, Mateusz and Serwata, Damian and Qi, Yu-Xuan
      and Jia, Mingshan and Musial, Katarzyna and Michalski, Rados{\l}aw
      and Br{\'o}dka, Piotr
    },
  journal={Big Data Mining And Analytics},
  volume={},
  number={},
  pages={1-13},
  year={2024},
  publisher={IEEE},
  doi = {10.26599/BDMA.2024.9020010},
  url={https://doi.org/10.26599/BDMA.2024.9020010},
}
```

Particularly if you used the functionality of simulating coexisting phenomena
in complex networks, please add the following reference:

```bibtex
@inproceedings{czuba2022coexisting,
  author={Czuba, Micha\l{} and Br\'{o}dka, Piotr},
  booktitle={9th International Conference on Data Science and Advanced Analytics (DSAA)},
  title={Simulating Spreading of Multiple Interacting Processes in Complex Networks},
  volume={},
  number={},
  pages={1-10},
  year={2022},
  month={oct},
  publisher={IEEE},
  address={Shenzhen, China},
  doi={10.1109/DSAA54385.2022.10032425},
  url={https://ieeexplore.ieee.org/abstract/document/10032425},
}
```

## Reporting bugs

Please report bugs on
[this](https://github.com/anty-filidor/network_diffusion/issues) board or by
sending a direct [e-mail](https://github.com/anty-filidor) to the main author.

## About us

This library is developed and maintained by
[Network Science Lab](https://networks.pwr.edu.pl/) from Politechnika
Wrocławska / Wrocław University of Science and Technology / Technische
Universität Breslau and external partners. For more information and updates,
please visit our [website](https://networks.pwr.edu.pl/) or
[GitHub](https://github.com/network-science-lab) page.
