# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time


class Timer:
    """Wall-clock timer reporting a moving average over recent intervals."""

    def __init__(self, history: int = 16):
        super().__init__()
        self.index = 0
        self.begin = None
        self.times = [0.0] * history
        self.history = history

    def start(self):
        self.begin = time.perf_counter()

    def stop(self):
        if self.begin is None:
            return

        t = time.perf_counter()
        elapsed = t - self.begin
        self.begin = t

        self.times[self.index % self.history] = elapsed
        self.index += 1

        return self.elapsed()

    def elapsed(self):
        idx = min(self.index, self.history)
        return 0 if idx == 0 else sum(self.times[:idx]) / idx

    def frequency(self):
        e = self.elapsed()
        return 0 if e == 0 else 1.0 / e
