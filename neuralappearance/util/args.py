# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import util


def parse_args():
    """Parse command line arguments."""

    default_config = util.get_root_dir() / 'configs' / 'default.json'

    default_asset_paths = util.get_default_asset_paths()

    parser = argparse.ArgumentParser(description='Train a neural material model')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=str(default_config),
        help=f'Path to the configuration file (default: {default_config})',
    )
    parser.add_argument('-o', '--outfolder', type=str, default=None, help='Path to output folder.')
    parser.add_argument(
        '-p',
        '--outfolder_policy',
        type=str,
        default='ask',
        help='Policy defining what to do when the outfolder exists. Options: [ask|append|overwrite].',
    )
    parser.add_argument('-m', '--message', type=str, help='Short description of experiment.')
    parser.add_argument(
        '-g', '--group', type=str, help='Short description of group of experiments.'
    )
    parser.add_argument(
        '-a',
        '--assets-paths',
        default=str(default_asset_paths),
        help='Semicolon-separated asset roots. Defaults to the repo root.',
    )
    parser.add_argument(
        '-v', '--view', action='store_true', help='Launch the job viewer after training completes.'
    )
    args = parser.parse_args()

    args.config = Path(args.config).resolve()

    if args.outfolder is None:
        args.outfolder = util.get_root_dir() / '.jobs' / util.get_unique_hash()
    else:
        args.outfolder = Path(args.outfolder)
    args.outfolder = args.outfolder.resolve()

    assert args.outfolder_policy in ['ask', 'append', 'overwrite']

    return args
