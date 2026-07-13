# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import sys

import commentjson as json


def deep_merge(base, override):
    """Recursively merge dictionaries, using content of override by default."""
    assert isinstance(base, dict)
    assert isinstance(override, dict)
    merged = copy.deepcopy(base)
    for key in override:
        if key in merged and isinstance(merged[key], dict) and isinstance(override[key], dict):
            merged[key] = deep_merge(merged[key], override[key])
        else:
            merged[key] = override[key]
    return merged


def load_config(filename):
    """Load a comment-enabled JSON config and recursively resolve base configs.

    ``base_config`` may name one file or a list. Paths are resolved relative to
    the config that declares them, bases are merged in order, and the current
    file overrides the merged result.
    """

    def load_file(filename: str):
        with open(filename) as ifile:
            config = json.load(ifile)
        if 'base_config' in config:
            configs = config.pop('base_config')
            configs = configs if isinstance(configs, list) else [configs]
            base = {}
            for current in configs:
                try:
                    owd = os.getcwd()
                    current = os.path.abspath(current)
                    os.chdir(os.path.dirname(current))
                    current = os.path.relpath(current, os.getcwd())
                    current = load_file(current)
                    os.chdir(owd)
                    base = deep_merge(base, current)
                except FileNotFoundError:
                    print('Could not locate config file', os.path.abspath(current))
                    sys.exit(1)
                except Exception as error:
                    print('Error while parsing config file', os.path.abspath(current))
                    print(error)
                    sys.exit(1)
            config = deep_merge(base, config)
        return config

    try:
        owd = os.getcwd()
        filename = os.path.abspath(filename)
        os.chdir(os.path.dirname(filename))
        filename = os.path.relpath(filename, os.getcwd())
        config = load_file(filename)
        os.chdir(owd)
        return config
    except FileNotFoundError:
        print('Could not locate config file', os.path.abspath(filename))
        sys.exit(1)
    except Exception as error:
        print('Error while parsing config file', os.path.abspath(filename))
        print(error)
        sys.exit(1)
