# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path


class LossSeries:
    """Simple JSON-backed loss series writer."""

    def __init__(self, outfolder: Path | str):
        self.outfolder = Path(outfolder)
        self.series: dict[str, dict[str, list[float]]] = {}

    def add(self, series_name: str, iteration: int | float, value: int | float) -> None:
        entries = self.series.setdefault(series_name, {'iteration': [], 'value': []})
        entries['iteration'].append(int(iteration))
        entries['value'].append(float(value))

    def flush(self) -> None:
        with (self.outfolder / 'loss_series.json').open('w') as ofile:
            json.dump(self.series, ofile, indent=2)
