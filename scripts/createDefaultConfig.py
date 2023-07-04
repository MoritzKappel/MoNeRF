#! /usr/bin/env python3
# -- coding: utf-8 --

"""createDefaultConfig.py: Creates a new config file with default values for a given method and dataset."""

from argparse import ArgumentParser
import os
import yaml
from pathlib import Path
import projectpath

with projectpath.context():
    import Framework
    from Logger import Logger
    from Methods.Base.Model import BaseModel
    from Methods.Base.Renderer import BaseRenderer
    from Methods.Base.Trainer import BaseTrainer
    from Implementations import Methods as MI
    from Implementations import Datasets as DI


def main():
    Logger.setMode(Logger.MODE_VERBOSE)
    # parse command line args
    parser: ArgumentParser = ArgumentParser(prog='createDefaultConfig')
    parser.add_argument(
        '-m', '--method', action='store', dest='method_name',
        metavar='method directory name', required=True,
        help='Name of the method you want to train. Name should match the directory in lib/methods.'
    )
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset_name',
        metavar='dataset name', required=True,
        help='Name of the dataset you want to train on. Name should match the python file in src/Datasets.'
    )
    parser.add_argument(
        '-a', '--all', action='store_true', dest='all_sequences',
        help='If set, creates a directory containing a config file for each sequence in the dataset.'
    )
    parser.add_argument(
        '-o', '--output', action='store', dest='output_filename',
        metavar='output config filename', required=True,
        help='Name of the generated config file (without extension).'
    )
    args = parser.parse_args()
    # create config with global defaults
    config: Framework.ConfigParameterList = Framework.ConfigParameterList(GLOBAL=Framework.getDefaultGlobalConfig())
    config.GLOBAL.METHOD_TYPE = args.method_name
    config.GLOBAL.DATASET_TYPE = args.dataset_name
    # add renderer, model and training parameters
    implementation_classes: tuple[BaseModel, BaseRenderer, BaseTrainer] = MI.getMethodClasses(args.method_name)
    config.MODEL = implementation_classes[0].getDefaultParameters()
    config.RENDERER = implementation_classes[1].getDefaultParameters()
    config.TRAINING = implementation_classes[2].getDefaultParameters()
    # add dataset parameters
    dataset_class = DI.getDatasetClass(args.dataset_name)
    config.DATASET = dataset_class.getDefaultParameters()
    # dump config into file
    output_path = Path(__file__).resolve().parents[1] / 'configs'
    dataset_path = None
    if args.all_sequences:
        output_path = output_path / args.output_filename
        dataset_path = Path(config.DATASET.PATH).parents[0]
        os.makedirs(str(output_path), exist_ok=True)
        if not dataset_path.is_dir():
            Logger.logError(f'failed to gather sequences from "{dataset_path}": directory not found')
            return
        config_file_names = [i.name for i in dataset_path.iterdir() if i.is_dir()]
    else:
        config_file_names = [args.output_filename]
    for config_file_name in config_file_names:
        config_file_path = output_path / f'{config_file_name}.yaml'
        try:
            config.TRAINING.MODEL_NAME = config_file_name
            if dataset_path is not None:
                config.DATASET.PATH = str(dataset_path / config_file_name)
            with open(config_file_path, 'w') as f:
                yaml.dump(Framework.ConfigParameterList.toDict(config), f, default_flow_style=False, indent=4, canonical=False, sort_keys=False)
                Logger.logInfo(f'configuration file successfully created: {config_file_path}')
        except IOError as e:
            Logger.logError(f'failed to create configuration file: "{e}"')


if __name__ == '__main__':
    main()
