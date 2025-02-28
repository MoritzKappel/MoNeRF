# MoNeRF: Fast Non-Rigid Radiance Fields from Monocularized Data

### | [Project Page](https://graphics.tu-bs.de/publications/kappel2024fast) | [Paper](https://arxiv.org/abs/2212.01368) | [Video](https://www.youtube.com/watch?v=1L73suob3bU) | [Dataset](https://nextcloud.mpi-klsb.mpg.de/index.php/s/EHtctQJZDWWcfqj) |

Official PyTorch Implementation of 'Fast Non-Rigid Radiance Fields from Monocularized Data'.

<img src='resources/banner.gif' width=1200>

## Changelog

- [28.02.25] Updated the code to be compatible with the official [NeRFICG](https://github.com/nerficg-project) project release!

## Getting Started

This project is built on the [NeRFICG](https://github.com/nerficg-project) framework. Before cloning this repository, ensure the framework is set up:

- Follow the instructions in the *Getting Started* section of the main [nerficg](https://github.com/nerficg-project/nerficg) repository (tested with commit __c8e258b__, PyTorch 2.5).
- After setting up the framework, navigate to the top level directory:
	```shell
	cd <Path/to/framework/>nerficg
	```
- also make sure to activate the correct conda environment
	```shell
	conda activate nerficg
	```
Now, you can directly add this project as an additional method:
- clone this repository to the *src/Methods/* directory:
	```shell
	git clone git@github.com:MoritzKappel/MoNeRF.git src/Methods/MoNeRF
	```
- install all dependencies and CUDA extensions for the new method using:
	```shell
	./scripts/install.py -m MoNeRF
	```

## Training and Inference

After setup, the *MoNeRF* method is fully compatible with all *NeRFICG* framework scripts in the *scripts/* directory. This includes config file generation (*defaultConfig.py*), training (*train.py*), inference and performance benchmarking (*inference.py*), metric calculation (*generateTables.py*), and live rendering via the GUI (*gui.py*).

For guidance and detailed instruction, please refer to the [main nerficg repository](https://github.com/nerficg-project/nerficg). 

## Dataset

To use our MMVA dataset, first [download](https://nextcloud.mpi-klsb.mpg.de/index.php/s/EHtctQJZDWWcfqj) the zipped dataset and unpack it to the *nerficg/dataset* directory.
Then copy the *MMVA.py* dataloader file to the *nerficg/src/Datasets* directory to make it available to the training and inference scripts.

## License and Citation

This project is licensed under the MIT license (see [LICENSE](LICENSE)).

If you use this code for your research projects, please consider a citation:
```bibtex
@article{kappel2024fast,
  title = {Fast Non-Rigid Radiance Fields from Monocularized Data},
  author = {Kappel, Moritz and Golyanik, Vladislav and Castillo, Susana  and Theobalt, Christian and Magnor, Marcus},
  journal = {{IEEE} Transactions on Visualization and Computer Graphics ({TVCG})},
  doi = {10.1109/{TVCG}.2024.3367431},
  pages = {1--12},
  month = {Feb},
  year = {2024}
}
```