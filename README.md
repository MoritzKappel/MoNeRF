
# MoNeRF: Fast Non-Rigid Radiance Fields from Monocularized Data

![Python](https://img.shields.io/static/v1?label=Python&message=3.11&color=success&logo=Python)&nbsp;![OS](https://img.shields.io/static/v1?label=OS&message=Linux/macOS&color=success&logo=Linux)&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

<img src='resources/banner.gif' width=1200>

<!-- ### [Website](TODO) | [ArXiv](TODO) | [Video](TODO) | [Data](TODO) <br> -->


Official PyTorch Implementation of 'Fast Non-Rigid Radiance Fields from Monocularized Data'

## Getting Started
- Clone this repository using:
```
git clone https://github.com/MoritzKappel/MoNeRF.git && cd MoNeRF
```

- Before running our code, you need to install all dependencies listed in *scripts/createCondaEnv.sh* under *# dependencies*, or create a new conda environment by executing the script:
```
./scripts/createCondaEnv.sh && conda activate monerf
```

- To install the necessary custom CUDA kernels, run:
```
./scripts/install.py -e VolumeRenderingV2
```

## Creating a Configuration File
Default configurations for the D-NeRF and MMVA datasets are available in the *configs/* directory.
To create a custom configuration file, run
```
./scripts/createDefaultConfig.py -m MoNeRF -d MMVA -o <my_config>
```
and edit the values in *configs/<my_config>.yaml* as needed.

## Dataset
You can manually download our MMVA dataset from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/EHtctQJZDWWcfqj), or run
```
./scripts/downloadDataset.sh
```
to automatically download and unpack the sequences to the *MoNeRF/dataset* directory.

## Training a New Model
To train a new model from a configuration file, run:
```
./scripts/train.py -c configs/<my_config>.yaml
```
The resulting images and model checkpoints will be saved to the *output* directory.


To train multiple models from a directory or list of configuration files, use the *scripts/sequentialTrain.py* script with the *-d* or *-c* flag respectively.


<!-- ## Training visualization
TODO wandb -->

<!-- ## Citation
If find this repository useful for your work, please consider citing our [paper](TODO) using the following BibTeX:

```
TODO
``` -->

<!-- ## Acknowledgments
This work was partially funded by the DFG (MA2555/15-1 ``Immersive Digital Reality'') and the ERC Consolidator Grant 4DRepLy (770784). -->

