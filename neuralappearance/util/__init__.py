# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from .args import parse_args
from .config import deep_merge, load_config
from .job import (
    create_job_summary,
    finalize_job_summary,
)
from .loss_series import LossSeries
from .math_helpers import cosine_falloff, udim_offset
from .print_to_file import PrintToFile
from .sample_generator import UniformSampleGenerator
from .timer import Timer
from .write_model_to_buffer import write_models_to_buffer


def get_root_dir():
    """Return path to the project root dir."""
    return Path(__file__).resolve().parent.parent.parent


def get_default_asset_paths():
    """Return default material asset search roots."""
    return str(get_root_dir().resolve())


def get_unique_hash():
    """Return a compact timestamp used as the default training-job directory."""
    now = datetime.now()
    return now.strftime('%m%d-%H%M-%S%f')[:-4]


def prepare_outfolder(outfolder: Path, mode: str = 'ask'):
    """Create output folder for a training job."""
    folder_exists = outfolder.is_dir() and len(list(outfolder.iterdir())) > 0

    if mode == 'append':
        assert folder_exists
        return
    elif mode == 'overwrite':
        if folder_exists:
            shutil.rmtree(outfolder, ignore_errors=False)
    elif folder_exists:
        print(f'Run folder "{outfolder}" is not empty. Erase and continue? [Y/n]', end=' ')
        answer = input().lower() or 'y'
        if answer != 'y':
            print('Exiting...')
            sys.exit()
        shutil.rmtree(outfolder, ignore_errors=False)

    print(f'Creating job folder {outfolder}')
    Path(outfolder).mkdir(parents=True, exist_ok=True)


def initialize_job(
    outfolder: Path,
    outfolder_policy: str,
    message: str,
    group: str,
    commit_hash: str | None = None,
):
    """Create job metadata, start log capture, and install signal handlers."""

    prepare_outfolder(outfolder, outfolder_policy)
    create_job_summary(outfolder, message, group, commit_hash)

    global _PRINT_TO_FILE
    _PRINT_TO_FILE = PrintToFile(outfolder / 'output.log', 'a')

    def on_signal(signal_number=None, stack_frame=None):
        finalize_job(outfolder=outfolder, signal_number=signal_number)
        raise InterruptedError()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    return outfolder


def finalize_job(outfolder: Path, signal_number=None):
    """Close log capture and record the job's final status."""

    finalize_start_time = time.perf_counter()

    # Handle "Interrupt" and "Terminate" signals.
    close_start_time = time.perf_counter()
    _PRINT_TO_FILE.close()
    close_elapsed = time.perf_counter() - close_start_time
    print(f'Close log time: {close_elapsed:.3f}s')

    status = 'Completed'
    if signal_number == signal.SIGINT:
        status = 'Interrupted'
    elif signal_number == signal.SIGTERM:
        status = 'Terminated'
    elif signal_number is not None:
        status = f'Received signal {signal_number}'

    if outfolder:
        summary_start_time = time.perf_counter()
        finalize_job_summary(outfolder, status)
        summary_elapsed = time.perf_counter() - summary_start_time
        print(f'Finalize job summary time: {summary_elapsed:.3f}s')

    total_elapsed = time.perf_counter() - finalize_start_time
    print(f'Finalize job total time: {total_elapsed:.3f}s')


__all__ = [
    'LossSeries',
    'PrintToFile',
    'Timer',
    'UniformSampleGenerator',
    'cosine_falloff',
    'create_job_summary',
    'deep_merge',
    'finalize_job',
    'finalize_job_summary',
    'get_default_asset_paths',
    'get_root_dir',
    'get_unique_hash',
    'initialize_job',
    'load_config',
    'parse_args',
    'prepare_outfolder',
    'udim_offset',
    'write_models_to_buffer',
]
