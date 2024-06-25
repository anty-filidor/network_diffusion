"""Setup script to produce a package from the code."""

from setuptools import Extension, find_packages, setup


def parse_requirements() -> list[str]:
    """Parse requirements from the txt file."""
    with open(file="env/pip_prod.txt", encoding="utf-8") as file:
        requirements = file.readlines()
    return requirements


def parse_readme() -> str:
    """Convert README to rst standard."""
    with open("README.md", encoding="utf-8") as file:
        long_description = file.read()
    return long_description


setup(
    long_description=parse_readme(),
    long_description_content_type="text/markdown",
    install_requires=parse_requirements(),
    ext_modules=[
        Extension(
            name="network_diffusion.tpn.cogsnet_lib",
            include_dirs=["c_modules"],
            sources=["c_modules/cogsnet_compute.c", "c_modules/cogsnet_lib.c"],
        )
    ],
    packages=find_packages(exclude=["*tests*"]),
)
