# Layout Model Wrappers

This repository is meant to contain several layout models, each with its own wrapper code. The purpose of these wrappers is to provide a unified way to use the layout models through a common parameter interface.

The layout models are included as submodules from the original authors’ repositories. The implemented layout models, as well as upcoming ones, are listed as follows

- [x] [HorizonNet - (CVPR'19)](https://github.com/sunset1995/HorizonNet)
- [ ] [HohoNet - (CVPR'21)](https://github.com/sunset1995/HoHoNet)
- [ ] [Led2Net - (CVPR'21)](https://github.com/fuenwang/LED2-Net)
- [x] [LGTNet - (CVPR'22)](https://github.com/zhigangjiang/LGT-Net)
- [ ] [DOPNet - (CVPR'23)](https://github.com/zhijieshen-bjtu/DOPNet)

## Installation

### Create a virtual environment
```sh 
conda create -n layout_models python=3.9
conda activate layout_models
```

### Install the package from the repository
```sh
pip install git+https://github.com/EnriqueSolarte/layout_models.git@257d87f7988884777dc4e54261742955028bbe96
```

### For installing this package in dev mode (for development)
```sh 
git clone git@github.com:EnriqueSolarte/layout_models.git
cd layout_models
git submodule update --init --recursive
pip install -e .
```
