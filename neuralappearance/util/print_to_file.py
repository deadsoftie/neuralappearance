# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import sys
from datetime import datetime


class _StreamWrapper:
    """Wrap either stdout or stderr, to also redirect to a log file."""

    def __init__(self, handle, stream, raw_print):
        self.handle = handle
        self.stream = stream
        self.raw_print = raw_print
        if stream is sys.stdout:
            self.level = 'STDOUT'
        elif stream is sys.stderr:
            self.level = 'STDERR'
        else:
            raise Exception(f'Unknown stream: {stream}, expected sys.stdout or sys.stderr.')

        self.ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

    def write(self, buf):
        """Write simultaneously to the output stream and the log file."""
        print(buf, file=self.stream, end='')
        for line in buf.rstrip().splitlines():
            line = self.ansi_escape.sub('', line)
            if self.raw_print:
                print(line, file=self.handle, flush=True)
            else:
                print(
                    '{date} {level:<5} {line}'.format(
                        date=datetime.now().strftime('[%Y-%m-%d %H:%M]'),
                        level=self.level,
                        line=line,
                    ),
                    file=self.handle,
                    flush=True,
                )

    def flush(self):
        """Expose the flushing mechanism of streams."""
        self.stream.flush()


class PrintToFile:
    """Wrap write calls to stdout/stderr and duplicate them in a log file.

    Usage:
    ------

    >>> with PrintToFile('output.log', 'w'):
    >>>     print('This goes to stdout and output.log')
    >>>     print('This goes to stderr and output.log', file=sys.stderr)
    >>> print('This goes only to stdout')

    Terminal output:
    This goes to stdout and output.log
    This goes to stderr and output.log
    This goes only to stdout

    Content of output.log:
    [2019-02-12 13:44] STDOUT This goes to stdout and output.log
    [2019-02-12 13:44] STDERR This goes to stderr and output.log

    """

    def __init__(self, filename, mode, raw_print=False):
        self.log_file = open(filename, mode)
        sys.stdout = _StreamWrapper(self.log_file, sys.stdout, raw_print)
        sys.stderr = _StreamWrapper(self.log_file, sys.stderr, raw_print)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        """Restore stdout and stderr, then close the log file."""
        sys.stdout = sys.stdout.stream
        sys.stderr = sys.stderr.stream
        self.log_file.close()
