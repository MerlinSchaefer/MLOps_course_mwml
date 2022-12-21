# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# setup.py
setup(
    name="tagifai",
    version=0.1,
    description="Classify machine learning projects.",
    author="Merlin Schaefer",
    author_email="None",
    url="https://madewithml.com/",
    python_requires=">=3.7",
    install_requires=[required_packages],
)
