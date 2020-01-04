from setuptools import setup

setup(
   name="network_diffusion",
   version="0.5.0",
   url="https://github.com/anty-filidor/network_diffusion",
   project_urls={
      "Documentation": "https://network-diffusion.readthedocs.io/en/latest/",
      "Code": "https://github.com/anty-filidor/network_diffusion",
   },
   license="MIT",
   description="Package to design and run diffusion phenomena processes in networks",
   author="Michał Czuba",
   author_email="michal.czuba.1995@gmail.com",
   packages=["network_diffusion"],
   install_requires=[
      'certifi == 2019.11.28',
      'cycler == 0.10.0',
      'decorator == 4.4.1',
      'kiwisolver == 1.1.0',
      'matplotlib == 3.1.1',
      'multipledispatch == 0.6.0',
      'networkx == 2.4',
      'numpy == 1.17.4',
      'olefile == 0.46',
      'pandas == 0.25.3',
      'Pillow == 6.2.1',
      'pyparsing == 2.4.5',
      'python - dateutil == 2.8.1',
      'pytz == 2019.3',
      'six == 1.13.0',
      'tornado == 6.0.3',
      'tqdm == 4.40.2'
         ],
)

