#! /usr/bin/env python3
# -- coding: utf-8 --

"""install.py: installs cuda extensions."""

import os
from argparse import ArgumentParser

import projectpath

with projectpath.context():
    from Logger import Logger
    from Implementations import CudaExtensions as CE


def main():
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    # parse arguments
    Logger.setMode(Logger.MODE_VERBOSE)
    parser: ArgumentParser = ArgumentParser(prog='Install')
    parser.add_argument(
        '-e', '--extension', action='store', dest='extension_name', default=None,
        metavar='cuda_extension_name', required=False,
        help='name of the cuda extension to be installed. Leave out to install all extensions'
    )
    args = parser.parse_args()

    if args.extension_name is not None:
        CE.installExtension(args.extension_name)
    else:
        CE.installAll()


if __name__ == '__main__':
    main()
