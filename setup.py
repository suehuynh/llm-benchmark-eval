from setuptools import find_packages, setup
from typing import List

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="anime-recsys",
    version="0.0.1",
    author="SueHuynh",
    packages= find_packages(),
    install_requires=requirements
)