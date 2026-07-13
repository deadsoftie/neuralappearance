# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import getpass
import hashlib
import json
import os
import sys
from datetime import datetime


def _summary_path(job_dir):
    return os.path.join(job_dir, 'summary.json')


def _write_job_summary(job_dir, job_summary):
    with open(_summary_path(job_dir), 'w', newline='\n') as ofile:
        json.dump(job_summary, ofile, indent=4)


def create_job_summary(job_dir, job_description, group_description, commit_hash=None):
    """Create a job summary according to the command line arguments."""

    # Git metadata is intentionally omitted to keep job creation inexpensive.

    if not group_description:
        group_description = 'default group'
    group_id = hashlib.md5(group_description.encode()).hexdigest()

    job_summary = {
        'description': job_description,
        'user': getpass.getuser(),
        'group_id': group_id,
        'group_description': group_description,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'commit': commit_hash,
        'cmd': ' '.join(sys.argv),
        'phases': [],
        'status': 'Running',
    }
    for key in ['group_id', 'group_description']:
        if not job_summary[key]:
            del job_summary[key]
    _write_job_summary(job_dir, job_summary)


def finalize_job_summary(job_dir, status):
    """Update the job summary with the end time and status."""
    with open(_summary_path(job_dir)) as ifile:
        job_summary = json.load(ifile)
    job_summary['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    # Reinsert ``status`` so it remains the final field in the JSON file.
    job_summary.pop('status')
    job_summary['status'] = status
    _write_job_summary(job_dir, job_summary)
