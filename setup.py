from setuptools import find_packages, setup

with open("./requirements.txt", "r") as f:
    requirements = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

setup(
    name="layout_models",
    version="0.1",
    packages=find_packages(),
    package_data={
                  "layout_models": ["model/**", "config/**"]
                  },
    install_requires=requirements,
    author="Enrique Solarte",
    author_email="enrique.solarte.pardo@gmail.com",
    description=("Layout estimation models (wrappers)."),
    license="BSD",
)
